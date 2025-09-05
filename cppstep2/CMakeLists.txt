cmake_minimum_required(VERSION 3.20)

project(BPE_TRAIN_STEP2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


add_library(bpe_train_step2 SHARED bpe_train_step2.cpp)
target_include_directories(bpe_train_step2 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(bpe_train_step2 PUBLIC MY_LIBRARY_EXPORT)

target_include_directories(bpe_train_step2 PUBLIC
    "${PROJECT_SOURCE_DIR}/emhash"
)

install(TARGETS bpe_train_step2
        EXPORT bpe_train_step2_export_targets
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

install(FILES "bpe_train_step2.h" 
        DESTINATION include
)

install(FILES "emhash/hash_table8.hpp" 
        DESTINATION include/emhash
)