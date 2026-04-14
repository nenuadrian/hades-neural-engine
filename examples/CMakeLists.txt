if(HNE_BUILD_TRAINING AND HNE_BUILD_INFERENCE)
    add_executable(train_cartpole
        cartpole/train_cartpole.cpp
    )
    target_link_libraries(train_cartpole PRIVATE hne_training hne_inference)
endif()
