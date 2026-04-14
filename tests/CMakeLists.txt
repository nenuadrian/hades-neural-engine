include(FetchContent)
FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
    GIT_SHALLOW TRUE)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Core tests (no LibTorch)
add_executable(hne_test_types test_types.cpp)
target_link_libraries(hne_test_types PRIVATE hne_core GTest::gtest_main)
add_test(NAME hne_types COMMAND hne_test_types)

# Training tests (require LibTorch)
if(HNE_BUILD_TRAINING)
    add_executable(hne_test_replay_buffer test_replay_buffer.cpp)
    target_link_libraries(hne_test_replay_buffer PRIVATE hne_training GTest::gtest_main)
    add_test(NAME hne_replay_buffer COMMAND hne_test_replay_buffer)

    add_executable(hne_test_ppo test_ppo.cpp)
    target_link_libraries(hne_test_ppo PRIVATE hne_training GTest::gtest_main)
    add_test(NAME hne_ppo COMMAND hne_test_ppo)
endif()

# Inference tests (require LibTorch)
if(HNE_BUILD_TRAINING AND HNE_BUILD_INFERENCE)
    add_executable(hne_test_inference test_inference.cpp)
    target_link_libraries(hne_test_inference PRIVATE hne_training hne_inference GTest::gtest_main)
    add_test(NAME hne_inference COMMAND hne_test_inference)
endif()
