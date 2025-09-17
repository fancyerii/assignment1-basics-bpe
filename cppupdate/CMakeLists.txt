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
add_executable(bpe_train_updater_omp_v7 bpe_train_updater_omp_v7.cpp)
add_executable(bpe_train_updater_omp_v8 bpe_train_updater_omp_v8.cpp)

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


add_executable(bpe_train_updater_fine_grained bpe_train_updater_fine_grained.cpp)

add_executable(bpe_train_updater_fine_grained_emhash8 bpe_train_updater_fine_grained_emhash8.cpp)

target_include_directories(bpe_train_updater_fine_grained_emhash8 PUBLIC
    "${PROJECT_SOURCE_DIR}/emhash"
)

add_executable(bpe_train_updater_fine_grained_absl bpe_train_updater_fine_grained_absl.cpp)
target_link_libraries(bpe_train_updater_fine_grained_absl PRIVATE absl::flat_hash_map absl::hash)


add_executable(bpe_train_updater_fine_grained_emhash8_set bpe_train_updater_fine_grained_emhash8_set.cpp)

target_include_directories(bpe_train_updater_fine_grained_emhash8_set PUBLIC
    "${PROJECT_SOURCE_DIR}/emhash"
)

add_executable(bpe_train_updater_fine_grained_emhash8_set9 bpe_train_updater_fine_grained_emhash8_set9.cpp)
target_include_directories(bpe_train_updater_fine_grained_emhash8_set9 PUBLIC
    "${PROJECT_SOURCE_DIR}/emhash"
)

target_link_libraries(bpe_train_updater_omp_v2 PRIVATE absl::flat_hash_map absl::hash)
target_link_libraries(bpe_train_updater_omp_v2_hash PRIVATE absl::flat_hash_map absl::hash)
target_link_libraries(bpe_train_updater_omp_v2_hash2 PRIVATE absl::flat_hash_map absl::hash)
target_link_libraries(bpe_train_updater_opt_absl PRIVATE absl::flat_hash_map absl::hash)


target_link_libraries(bpe_train_updater_omp_v4 PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(bpe_train_updater_omp_v4_debug PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(bpe_train_updater_omp_v5 PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(bpe_train_updater_omp_v6 PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(bpe_train_updater_omp_v7 PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(bpe_train_updater_omp_v8 PRIVATE OpenMP::OpenMP_CXX)
#add_custom_command(
#    TARGET bpe_train_updater
#    POST_BUILD
     # 调用系统 cp 命令来保留符号链接，这需要你的系统支持
#    COMMAND cp -a "${CMAKE_SOURCE_DIR}/data" "${CMAKE_BINARY_DIR}/data"
#)

add_executable(test_unordered_map_traverse_speed test_unordered_map_traverse_speed.cpp)
add_executable(bpe_train_updater_max_debug bpe_train_updater_max_debug.cpp)
add_executable(bpe_train_updater_omp_v3_max_debug bpe_train_updater_omp_v3_max_debug.cpp)