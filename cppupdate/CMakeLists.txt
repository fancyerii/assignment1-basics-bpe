cmake_minimum_required(VERSION 3.20)

# 定义项目名称
project(BPE_TRAIN_UPDATER LANGUAGES CXX)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


add_subdirectory(abseil-cpp)

find_package(OpenMP REQUIRED)


# 添加可执行文件
add_executable(bpe_train_updater bpe_train_updater.cpp)
add_executable(bpe_train_updater_omp_v1 bpe_train_updater_omp_v1.cpp)
add_executable(bpe_train_updater_omp_v2 bpe_train_updater_omp_v2.cpp)
add_executable(bpe_train_updater_omp_v2_debug bpe_train_updater_omp_v2_debug.cpp)
add_executable(bpe_train_updater_omp_v3 bpe_train_updater_omp_v3.cpp)
add_executable(bpe_train_updater_omp_v4 bpe_train_updater_omp_v4.cpp)
add_executable(bpe_train_updater_omp_v4_debug bpe_train_updater_omp_v4_debug.cpp)
add_executable(bpe_train_updater_omp_v5 bpe_train_updater_omp_v5.cpp)
add_executable(bpe_train_updater_omp_v6 bpe_train_updater_omp_v6.cpp)

add_executable(bpe_train_updater_hash bpe_train_updater_hash.cpp)
add_executable(bpe_train_updater_omp_v1_hash bpe_train_updater_omp_v1_hash.cpp)
add_executable(bpe_train_updater_omp_v2_hash bpe_train_updater_omp_v2_hash.cpp)
add_executable(bpe_train_updater_omp_v2_hash2 bpe_train_updater_omp_v2_hash2.cpp)
add_executable(bpe_train_updater_omp_v3_hash bpe_train_updater_omp_v3_hash.cpp)
#add_executable(jsontest jsontest.cpp)
add_executable(bpe_train_updater_opt bpe_train_updater_opt.cpp)
add_executable(bpe_train_updater_opt_absl bpe_train_updater_opt_absl.cpp)
add_executable(bpe_train_updater_opt_emhash8 bpe_train_updater_opt_emhash8.cpp)
target_include_directories(bpe_train_updater_opt_emhash8 PUBLIC
    "${PROJECT_SOURCE_DIR}/emhash"
)

add_executable(bpe_train_updater_emhash8 bpe_train_updater_emhash8.cpp)
target_include_directories(bpe_train_updater_emhash8 PUBLIC
    "${PROJECT_SOURCE_DIR}/emhash"
)

add_executable(bpe_train_updater_opt_emhash8_hash bpe_train_updater_opt_emhash8_hash.cpp)
target_include_directories(bpe_train_updater_opt_emhash8_hash PUBLIC
    "${PROJECT_SOURCE_DIR}/emhash"
)

target_include_directories(bpe_train_updater_omp_v6 PUBLIC
    "${PROJECT_SOURCE_DIR}/emhash"
)

add_executable(omp_test omp_test.cpp)
target_link_libraries(omp_test PRIVATE OpenMP::OpenMP_CXX)

add_executable(test_emhash test_emhash.cpp)

add_executable(bpe_train_updater_fine_grained bpe_train_updater_fine_grained.cpp)

add_executable(bpe_train_updater_fine_grained_emhash8 bpe_train_updater_fine_grained_emhash8.cpp)

target_include_directories(bpe_train_updater_fine_grained_emhash8 PUBLIC
    "${PROJECT_SOURCE_DIR}/emhash"
)

add_executable(bpe_train_updater_fine_grained_absl bpe_train_updater_fine_grained_absl.cpp)
target_link_libraries(bpe_train_updater_fine_grained_absl PRIVATE absl::flat_hash_map absl::hash)

target_include_directories(test_emhash PUBLIC
    "${PROJECT_SOURCE_DIR}/emhash"
)

target_link_libraries(bpe_train_updater_omp_v2 PRIVATE absl::flat_hash_map absl::hash)
target_link_libraries(bpe_train_updater_omp_v2_hash PRIVATE absl::flat_hash_map absl::hash)
target_link_libraries(bpe_train_updater_omp_v2_hash2 PRIVATE absl::flat_hash_map absl::hash)
target_link_libraries(bpe_train_updater_opt_absl PRIVATE absl::flat_hash_map absl::hash)
#add_executable(hello_world hello_world.cc)
#target_link_libraries(hello_world absl::strings)
#target_compile_options(bpe_train_updater_omp_v2 PRIVATE -pg)
#target_link_options(bpe_train_updater_omp_v2 PRIVATE -pg) 

#target_include_directories(bpe_train_updater PRIVATE ${CMAKE_SOURCE_DIR})
#target_include_directories(bpe_train_updater_omp_v1 PRIVATE ${CMAKE_SOURCE_DIR})
#target_include_directories(bpe_train_updater_omp_v2 PRIVATE ${CMAKE_SOURCE_DIR})
#target_include_directories(jsontest PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(bpe_train_updater_omp_v4 PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(bpe_train_updater_omp_v4_debug PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(bpe_train_updater_omp_v5 PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(bpe_train_updater_omp_v6 PRIVATE OpenMP::OpenMP_CXX)
# 设置编译选项，例如开启所有警告
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
add_custom_command(
    TARGET bpe_train_updater
    POST_BUILD
    # 调用系统 cp 命令来保留符号链接，这需要你的系统支持
    COMMAND cp -a "${CMAKE_SOURCE_DIR}/data" "${CMAKE_BINARY_DIR}/data"
)

# add_library(bpe_train_step2 SHARED bpe_train_step2.cpp)
# target_include_directories(bpe_train_step2 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# target_compile_definitions(bpe_train_step2 PUBLIC MY_LIBRARY_EXPORT)

# install(TARGETS bpe_train_step2
#         EXPORT bpe_train_step2_export_targets
#         RUNTIME DESTINATION bin
#         LIBRARY DESTINATION lib
#         ARCHIVE DESTINATION lib)


# # install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
# #         DESTINATION include/bpe_train_step2
# #         FILES_MATCHING PATTERN "*.h")

# install(FILES "bpe_train_step2.h" 
#         DESTINATION include
# )

# install(FILES "emhash/hash_table8.hpp" 
#         DESTINATION include/emhash
# )