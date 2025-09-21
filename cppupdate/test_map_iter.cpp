#include <iostream>
#include <unordered_map>
#include <random>
#include <chrono>
#include <utility>

int main(){
    // 1. 设置随机数生成器
    // 使用当前时间作为种子，确保每次运行程序生成的序列都不同
    //unsigned seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::mt19937 gen(1234); // Mersenne Twister 引擎，高质量的伪随机数生成器

    // 2. 定义随机数分布范围
    // 键（key）和值（value）的生成范围
    std::uniform_int_distribution<> key_dist(1, 1000);   // 键的范围是 [1, 1000]
    std::uniform_int_distribution<> val_dist(1, 10000);  // 值的范围是 [1, 10000]

    // 3. 创建一个空的 std::unordered_map
    std::unordered_map<int, int> random_map;
    random_map.reserve(20);

    // 4. 填充 map，生成指定数量的随机键值对
    const int num_elements = 10;
    for (int i = 0; i < num_elements; ++i) {
        int random_key = key_dist(gen);
        int random_value = val_dist(gen);
        random_map[random_key] = random_value;
        std::cout << "insert: " << random_key << "->" << random_value << std::endl;
    }

    // 5. 打印生成的 map

    std::cout << "生成的随机 unordered_map 包含 " << random_map.size() << " 个元素:" << std::endl;
    std::cout << "for遍历" << std::endl;
    for (const auto& pair : random_map) {
        //vec1.push_back({pair.first, pair.second});
        std::cout << "\t" << pair.first << "," << pair.second << std::endl;
    }
    std::cout << "iterator" << std::endl;
    for (auto it = random_map.begin(); it != random_map.end(); ++it) {
        std::cout << "\t" << it->first << "," << it->second << std::endl;
    }
    std::cout << "bucket api" << std::endl;
    size_t num_buckets = random_map.bucket_count();
    for (size_t i = 0, j = 0; i < num_buckets; ++i) {
        std::cout << "bucket " << i << std::endl;
        for (auto it = random_map.begin(i); it != random_map.end(i); ++it) {
            std::cout << "\t" << it->first << "," << it->second << std::endl;
        }
    }

}