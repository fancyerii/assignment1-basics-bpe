#include "max_heap.h"
#include <iostream>
#include <vector>
#include <utility>
#include <random>
#include <chrono>
#include <algorithm>

// 辅助函数：生成一个随机的 HeapItem 对象
HeapItem generate_random_heap_item(std::mt19937& rng) {
    HeapItem item;
    
    // 生成随机的 count
    std::uniform_int_distribution<int> count_dist(0, 100000);
    item.count = count_dist(rng);
    
    // 生成随机的 pair
    std::uniform_int_distribution<int> pair_dist(0, 100000);
    item.pair = {pair_dist(rng), pair_dist(rng)};

    // 随机生成 pair_string
    std::uniform_int_distribution<int> vec_size_dist(1, 5); // 随机生成 vector 的大小
    
    item.pair_string.resize(2);
    for (int i = 0; i < 2; ++i) {
        int inner_size = vec_size_dist(rng);
        item.pair_string[i].resize(inner_size);
        for (int j = 0; j < inner_size; ++j) {
            item.pair_string[i][j] = pair_dist(rng);
        }
    }
    
    return item;
}

// 主函数：生成随机的 std::vector<HeapItem>
std::vector<HeapItem> generate_random_heap_vector(size_t size, int seed_arg) {
    unsigned seed;
    if(seed_arg == -1){
        seed = std::chrono::system_clock::now().time_since_epoch().count();
    }else{
        seed = seed_arg;
    }
    
    std::mt19937 rng(seed);
    
    std::vector<HeapItem> random_vector;
    random_vector.reserve(size);

    for (size_t i = 0; i < size; ++i) {
        random_vector.push_back(generate_random_heap_item(rng));
    }
    
    return random_vector;
}

bool vec_equal(const std::vector<HeapItem>& items, const std::vector<HeapItem>& items2){
    bool equal = false;
    if(items.size() == items2.size()){
        equal = std::equal(items.begin(), items.end(), items2.begin(),
                [](const HeapItem& a, const HeapItem& b) {
                return a.count == b.count && a.pair == b.pair && a.pair == b.pair;
            });
    }
    return equal;
}
bool item_equal(const HeapItem& a, const HeapItem& b){
    return a.count == b.count && a.pair == b.pair && a.pair == b.pair;
}

int main(int argc, char * argv[]){
    if(argc < 3){
        std::cout << "usage: test_max_heap loop_count heap_size [seed]" << std::endl;
        return -1;
    }
    int loop_count = std::stoi(argv[1]); 
    int heap_size =std::stoi(argv[2]);
 
    int seed = -1;
 
    if(argc > 3){
        seed = std::stoi(argv[3]);
    }
    std::cout << "loop_count: " << loop_count << ", heap_size: " <<heap_size << std::endl;
    std::cout << "seed: " << seed <<std::endl;

    for(int i = 0; i < loop_count; ++i){
        std::vector<HeapItem> items = generate_random_heap_vector(heap_size, seed);
        std::vector<HeapItem> items2(items);
        heapify(items);
        std::make_heap(items2.begin(), items2.end());
        bool equal = vec_equal(items, items2);
        if(!equal){
            std::cout << "bug, seed: " << seed << ", i: " << i << std::endl;
            return -1;
        }
    }

    //test speed
    auto time1 = std::chrono::steady_clock::duration::zero();
    auto time2 = std::chrono::steady_clock::duration::zero();
    
    for(int i = 0; i < loop_count; ++i){
        std::vector<HeapItem> items = generate_random_heap_vector(heap_size, seed);
        std::vector<HeapItem> items2(items);
        auto start_time = std::chrono::high_resolution_clock::now();
        heapify(items);
        auto end_time = std::chrono::high_resolution_clock::now();
        time1 += (end_time - start_time);
        start_time = std::chrono::high_resolution_clock::now();
        std::make_heap(items2.begin(), items2.end());
        end_time = std::chrono::high_resolution_clock::now();
        time2 += (end_time - start_time);
    }    

    std::cout << "heapify time. time1: " << std::chrono::duration_cast<std::chrono::milliseconds>(time1).count()
              << ", time2: " << std::chrono::duration_cast<std::chrono::milliseconds>(time2).count() << std::endl;


    for(int i = 0; i < loop_count; ++i){
        std::vector<HeapItem> items = generate_random_heap_vector(heap_size, seed);
        std::vector<HeapItem> items2(items);
        auto split = items.begin() + items.size() * 9 / 10;
        std::vector<HeapItem> items_first(items.begin(), split); 
        std::vector<HeapItem> items_second(split, items.end());
        
        auto split2 = items2.begin() + items2.size() * 9 / 10;
        std::vector<HeapItem> items2_first(items2.begin(), split2); 
        std::vector<HeapItem> items2_second(split2, items2.end());        
        
        heapify(items_first);
        std::make_heap(items2_first.begin(), items2_first.end());
        for(const auto& item : items_second){
            heappush(items_first, item);
            auto item_pop = heappop(items_first);
            
            items2_first.push_back(item);
            std::push_heap(items2_first.begin(), items2_first.end());

            std::pop_heap(items2_first.begin(), items2_first.end());
            auto item_pop2 = items2_first.back();
            items2_first.pop_back();
            if(!vec_equal(items_first, items2_first)){
                std::cout << "bug, seed: " << seed << ", i: " << i << std::endl;
                return -1;
            }
            if(!item_equal(item_pop, item_pop2)){
                std::cout << "bug, seed: " << seed << ", i: " << i << std::endl;
                return -1;                
            }
        }
    }

    time1 = std::chrono::steady_clock::duration::zero();
    time2 = std::chrono::steady_clock::duration::zero();
    for(int i = 0; i < loop_count; ++i){
        std::vector<HeapItem> items = generate_random_heap_vector(heap_size, seed);
        std::vector<HeapItem> items2(items);
        auto split = items.begin() + items.size() * 9 / 10;
        std::vector<HeapItem> items_first(items.begin(), split); 
        std::vector<HeapItem> items_second(split, items.end());
        
        auto split2 = items2.begin() + items2.size() * 9 / 10;
        std::vector<HeapItem> items2_first(items2.begin(), split2); 
        std::vector<HeapItem> items2_second(split2, items2.end());        
        
        heapify(items_first);
        std::make_heap(items2_first.begin(), items2_first.end());
        for(const auto& item : items_second){
            auto start_time = std::chrono::high_resolution_clock::now();
            heappush(items_first, item);
            auto item_pop = heappop(items_first);
            auto end_time = std::chrono::high_resolution_clock::now();
            time1 += (end_time - start_time);

            start_time = std::chrono::high_resolution_clock::now();
            items2_first.push_back(item);
            std::push_heap(items2_first.begin(), items2_first.end());

            std::pop_heap(items2_first.begin(), items2_first.end());
            auto item_pop2 = items2_first.back();
            items2_first.pop_back();
            end_time = std::chrono::high_resolution_clock::now();
            time2 += (end_time - start_time);

            if(!vec_equal(items_first, items2_first)){
                std::cout << "bug, seed: " << seed << ", i: " << i << std::endl;
                return -1;
            }
            if(!item_equal(item_pop, item_pop2)){
                std::cout << "bug, seed: " << seed << ", i: " << i << std::endl;
                return -1;                
            }
        }
    }
    std::cout << "push/pop time. time1: " << std::chrono::duration_cast<std::chrono::milliseconds>(time1).count()
              << ", time2: " << std::chrono::duration_cast<std::chrono::milliseconds>(time2).count() << std::endl;


    time1 = std::chrono::steady_clock::duration::zero();
    time2 = std::chrono::steady_clock::duration::zero();
    
    for(int i = 0; i < loop_count; ++i){
        std::vector<HeapItem> items = generate_random_heap_vector(heap_size, seed);
        std::vector<HeapItem> items2(items);
        heapify(items);
        std::sort(items2.begin(), items2.end(), 
            [](const HeapItem& a, const HeapItem& b){
                return b < a;
            });
        for(int j = 0; j < 100; ++j){
            auto item = heappop(items);
            if(!item_equal(item, items2[j])){
                std::cout << "bug, seed: " << seed << ", i: " << i << ", j: " << j << std::endl;
                return -1;                    
            }
        }
    }      
}