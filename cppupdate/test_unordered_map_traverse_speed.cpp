#include <iostream>
#include <unordered_map>
#include <random>
#include <chrono>
#include <vector>
#include <utility>


int main(int argc, char * argv[]){
    if(argc < 3){
        std::cout << "usage: test_unordered_map_traverse_speed loop_count map_size [key_range, seed]" << std::endl;
        return -1;
    }
    int loop_count = std::stoi(argv[1]); 
    int map_size =std::stoi(argv[2]);
    int key_range = 10000;
    int seed = 1234;
    if(argc > 3){
        key_range = std::stoi(argv[3]);
    }
    if(argc > 4){
        seed = std::stoi(argv[4]);
    }
    std::cout << "loop_count: " << loop_count << ", map_size: " <<map_size << std::endl;
    std::cout << "key_range: " <<key_range << ", seed: " << seed <<std::endl;
    std::mt19937 gen(seed);


    std::uniform_int_distribution<> key_dist(1, key_range); 
    std::uniform_int_distribution<> val_dist(1, key_range); 

    auto iter_time = std::chrono::steady_clock::duration::zero();
    auto bucket_time = std::chrono::steady_clock::duration::zero();
    for(int loop = 0; loop < loop_count; ++loop){
        std::unordered_map<int, int> random_map;

        for (int i = 0; i < map_size; ++i) {
            int random_key = key_dist(gen);
            int random_value = val_dist(gen);
            random_map[random_key] = random_value;
        }
        auto start1 = std::chrono::steady_clock::now();
        int sum1 = 0;
        for(const auto& pair : random_map){
            sum1 += (pair.first + pair.second);
        }
        auto end1 = std::chrono::steady_clock::now();
        iter_time += (end1 - start1);
        auto start2 = std::chrono::steady_clock::now();
        int sum2 = 0;
        size_t num_buckets = random_map.bucket_count();
        for (size_t i = 0, j = 0; i < num_buckets; ++i) {
            for (auto it = random_map.begin(i); it != random_map.end(i); ++it) {
                sum2 += (it->first + it->second);
            }
        }       
        if(sum1 != sum2){
            std::cout << "bug!" << std::endl;

            return -1;
        }
        auto end2 = std::chrono::steady_clock::now(); 
        bucket_time += (end2 - start2);
    }
    auto iter_duration = std::chrono::duration_cast<std::chrono::milliseconds>(iter_time); 
    auto bucket_duration = std::chrono::duration_cast<std::chrono::milliseconds>(bucket_time);
    std::cout << "iter_duration:   " << iter_duration.count()   << "ms" << std::endl;
    std::cout << "bucket_duration: " << bucket_duration.count() << "ms" << std::endl;
    return 0;
}