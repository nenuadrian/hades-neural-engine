cmake_minimum_required(VERSION 3.20)
project(HadesNeuralEngine VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(HNE_BUILD_TRAINING "Build training components (requires LibTorch)" OFF)
option(HNE_BUILD_INFERENCE "Build inference runtime (requires LibTorch CPU)" OFF)
option(HNE_BUILD_IMGUI_WIDGETS "Build ImGui training widgets (consumer must provide ImGui)" OFF)
option(HNE_BUILD_TESTS "Build unit tests" OFF)
option(HNE_BUILD_EXAMPLES "Build example programs" OFF)

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

# ── Tests ───────────────────────────────────────────────────────────────────

if(HNE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ── Examples ────────────────────────────────────────────────────────────────

if(HNE_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
