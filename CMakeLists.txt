cmake_minimum_required(VERSION 3.20)
project(HadesNeuralEngine VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(HNE_BUILD_TRAINING "Build training components (requires LibTorch)" OFF)
option(HNE_BUILD_INFERENCE "Build inference runtime (requires LibTorch CPU)" OFF)
option(HNE_BUILD_IMGUI_WIDGETS "Build ImGui training widgets (consumer must provide ImGui)" OFF)
option(HNE_BUILD_WANDB "Build the Weights & Biases callback + panel (requires libcurl)" OFF)
option(HNE_BUILD_TESTS "Build unit tests" OFF)
option(HNE_BUILD_EXAMPLES "Build example programs" OFF)
option(HNE_AUTO_DETECT_TORCH "Auto-detect pip-installed PyTorch via `python -c 'import torch'`" ON)

# ── Dependencies ────────────────────────────────────────────────────────────

# nlohmann/json — required, skip fetch if consumer already provides it
if(NOT TARGET nlohmann_json::nlohmann_json)
    include(FetchContent)
    set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
    set(JSON_Install OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.3
        GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(nlohmann_json)
endif()

# LibTorch — only when training or inference is enabled
if(HNE_BUILD_TRAINING OR HNE_BUILD_INFERENCE)
    # Auto-detect pip-installed PyTorch unless the user already supplied a
    # Torch_DIR or a CMAKE_PREFIX_PATH that resolves Torch themselves.
    if(HNE_AUTO_DETECT_TORCH AND NOT Torch_DIR)
        set(_hne_torch_candidates "")

        # 1. Try the `python` / `python3` on PATH first (picks up active venv / conda env).
        find_program(_hne_py_path NAMES python3 python)
        if(_hne_py_path)
            list(APPEND _hne_torch_candidates "${_hne_py_path}")
        endif()

        # 2. Fall back to CMake's FindPython3.
        find_package(Python3 COMPONENTS Interpreter QUIET)
        if(Python3_Interpreter_FOUND)
            list(APPEND _hne_torch_candidates "${Python3_EXECUTABLE}")
        endif()

        list(REMOVE_DUPLICATES _hne_torch_candidates)

        foreach(_py IN LISTS _hne_torch_candidates)
            execute_process(
                COMMAND "${_py}" -c
                    "import torch,sys; sys.stdout.write(torch.utils.cmake_prefix_path)"
                OUTPUT_VARIABLE _hne_torch_prefix
                RESULT_VARIABLE _hne_torch_rc
                ERROR_QUIET)
            if(_hne_torch_rc EQUAL 0 AND _hne_torch_prefix)
                message(STATUS
                    "HNE: auto-detected LibTorch via ${_py}: ${_hne_torch_prefix}")
                list(PREPEND CMAKE_PREFIX_PATH "${_hne_torch_prefix}")
                break()
            endif()
        endforeach()

        if(NOT _hne_torch_prefix)
            message(STATUS
                "HNE: no Python with `import torch` found. Pass "
                "-DCMAKE_PREFIX_PATH=/path/to/libtorch or "
                "`pip install torch` in the active environment.")
        endif()
    endif()

    find_package(Torch REQUIRED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
endif()

# ── hne_core (always built, no LibTorch) ────────────────────────────────────

add_library(hne_core STATIC
    src/core/environment.cpp
    src/core/types.cpp
)
target_include_directories(hne_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(hne_core PUBLIC nlohmann_json::nlohmann_json)

# ── hne_training (optional, requires LibTorch) ──────────────────────────────

if(HNE_BUILD_TRAINING)
    add_library(hne_training STATIC
        src/core/replay_buffer.cpp
        src/algorithms/ppo.cpp
        src/training/trainer.cpp
        src/training/checkpoint.cpp
        src/training/metrics.cpp
    )
    target_compile_definitions(hne_training PUBLIC HNE_TRAINING)
    target_link_libraries(hne_training PUBLIC hne_core "${TORCH_LIBRARIES}")
    target_include_directories(hne_training PUBLIC "${TORCH_INCLUDE_DIRS}")
endif()

# ── hne_inference (optional, requires LibTorch for TorchScript) ─────────────

if(HNE_BUILD_INFERENCE)
    add_library(hne_inference STATIC
        src/inference/runtime.cpp
        src/inference/batch_evaluator.cpp
    )
    target_compile_definitions(hne_inference PUBLIC HNE_INFERENCE)
    target_link_libraries(hne_inference PUBLIC hne_core "${TORCH_LIBRARIES}")
    target_include_directories(hne_inference PUBLIC "${TORCH_INCLUDE_DIRS}")
endif()

# ── hne_imgui (optional, consumer provides ImGui target) ────────────────────

if(HNE_BUILD_IMGUI_WIDGETS)
    add_library(hne_imgui STATIC
        src/imgui/training_widgets.cpp
    )
    target_link_libraries(hne_imgui PUBLIC hne_core)
    # Consumer must link their own ImGui target alongside hne_imgui.
    # We only need the headers at compile time.
    if(TARGET imgui_lib)
        target_link_libraries(hne_imgui PUBLIC imgui_lib)
    elseif(TARGET imgui)
        target_link_libraries(hne_imgui PUBLIC imgui)
    endif()
endif()

# ── hne_wandb (optional, requires libcurl) ──────────────────────────────────

if(HNE_BUILD_WANDB)
    if(NOT HNE_BUILD_TRAINING)
        message(FATAL_ERROR
            "HNE_BUILD_WANDB requires HNE_BUILD_TRAINING (the callback derives "
            "from ITrainerCallback).")
    endif()
    find_package(CURL REQUIRED)

    set(_hne_wandb_sources
        src/wandb/client.cpp
        src/wandb/callback.cpp
    )
    if(HNE_BUILD_IMGUI_WIDGETS)
        list(APPEND _hne_wandb_sources src/imgui/wandb_panel.cpp)
    endif()

    add_library(hne_wandb STATIC ${_hne_wandb_sources})
    target_compile_definitions(hne_wandb PUBLIC HNE_WANDB)
    target_link_libraries(hne_wandb PUBLIC hne_training CURL::libcurl)

    if(HNE_BUILD_IMGUI_WIDGETS)
        # Same ImGui target lookup pattern as hne_imgui — the consumer is
        # expected to expose `imgui` or `imgui_lib`.
        if(TARGET imgui_lib)
            target_link_libraries(hne_wandb PUBLIC imgui_lib)
        elseif(TARGET imgui)
            target_link_libraries(hne_wandb PUBLIC imgui)
        endif()
    endif()
endif()

# ── Tests ───────────────────────────────────────────────────────────────────

if(HNE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ── Examples ────────────────────────────────────────────────────────────────

if(HNE_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
