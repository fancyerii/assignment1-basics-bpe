#include "bpe_train_step2.h"

#include <iostream>
#include <sstream>
#include <chrono>

// 自定义的三元比较结果枚举
enum class ComparisonResult {
    Less,
    Equal,
    Greater
};

// 实现高效的三元比较函数
ComparisonResult three_way_compare(const std::vector<int>& a, const std::vector<int>& b) {
    // 1. 获取较小的 vector 的大小
    size_t min_size = std::min(a.size(), b.size());

    // 2. 逐个元素进行比较
    for (size_t i = 0; i < min_size; ++i) {
        if (a[i] < b[i]) {
            return ComparisonResult::Less;
        }
        if (a[i] > b[i]) {
            return ComparisonResult::Greater;
        }
    }

    // 3. 如果所有共同元素都相同，则比较大小
    if (a.size() < b.size()) {
        return ComparisonResult::Less;
    }
    if (a.size() > b.size()) {
        return ComparisonResult::Greater;
    }

    // 4. 如果大小和所有元素都相同
    return ComparisonResult::Equal;
}

void fine_grained_pair_counter_diff(std::unordered_set<int>& affected_words,
                                    std::unordered_map<int, std::vector<int>> & wordid_encodings,
                                    const std::unordered_map<int, long long> & wordid_counts,
                                    const std::pair<int, int>& merge_pair,
                                    int new_id,
                                    std::unordered_map<std::pair<int, int>, std::unordered_set<int>, pair_hash> & pair_wordids,
                                    std::unordered_map<std::pair<int, int>, int, pair_hash> & diff_pairs,
                                    std::unordered_set<std::pair<int, int>, pair_hash> & new_pairs){

    for(int wordid : affected_words){
        const auto& word_tokens = wordid_encodings[wordid];
        auto& wc = wordid_counts.at(wordid);  
        
        int idx = 0;
        std::unordered_set<std::pair<int, int>, pair_hash> unaffected_pairs;

        int token_size_minus_1 = word_tokens.size() - 1;
        int token_size = word_tokens.size();
        int first_idx = -1;
        while(idx < token_size_minus_1){
            if(word_tokens[idx] == merge_pair.first &&
               word_tokens[idx + 1] == merge_pair.second){
                first_idx = idx;
                break;
            }
            ++idx;
        }
        if(first_idx == -1){
            std::stringstream ss;
            ss << "bug (" << merge_pair.first << "," << merge_pair.second
                      << "), " << wordid << ", [";
            for(auto token : word_tokens){
                ss << token << " ";
            }
            ss << "]" << std::endl;
            throw std::runtime_error(ss.str());
        }

        idx = word_tokens.size() - 2;
        int last_idx = -1;
        while(idx > first_idx + 1){
            if(word_tokens[idx] == merge_pair.first &&
               word_tokens[idx + 1 == merge_pair.second]){
                last_idx = idx;
                break;
            }
            --idx;
        }
        if(last_idx == -1){
            last_idx = first_idx;
        }

        int start_idx = std::max(0, first_idx - 1);
        int end_idx = std::min(last_idx + 3, token_size);

        for(int i = 0; i < start_idx; ++i){
            unaffected_pairs.emplace(word_tokens[i], word_tokens[i + 1]);
        }

        for(int i = end_idx - 1; i < token_size_minus_1; ++i){
            unaffected_pairs.emplace(word_tokens[i], word_tokens[i + 1]);
        }

        // c++ 20 support span, but we use only c++ 17
        const int * affected_tokens = word_tokens.data() + start_idx;
        int affected_tokens_len = end_idx - start_idx;

        for(int i = 0; i < affected_tokens_len - 1; ++i){
            std::pair<int, int> old_pair(affected_tokens[i], affected_tokens[i + 1]);
            diff_pairs[old_pair] -= wc;
            if (unaffected_pairs.find(old_pair) == unaffected_pairs.end()) {
                pair_wordids[old_pair].erase(wordid);
            }
        }

        std::vector<int> new_tokens;
        std::vector<int> all_new_tokens;
        for(int i = 0; i < start_idx; ++i){
            all_new_tokens.push_back(word_tokens[i]);
        }

        int i = 0;

        while(i < affected_tokens_len){
            if(i < affected_tokens_len - 1 &&
                    affected_tokens[i] == merge_pair.first &&
                    affected_tokens[i + 1] == merge_pair.second){
                new_tokens.push_back(new_id);
                all_new_tokens.push_back(new_id);
                i += 2;
            }else{
                new_tokens.push_back(affected_tokens[i]);
                all_new_tokens.push_back(affected_tokens[i]);
                ++i;
            }
        }

        for(int i = end_idx; i < token_size; ++i){
            all_new_tokens.push_back(word_tokens[i]);
        }

        wordid_encodings[wordid] = std::move(all_new_tokens);

        for(int i = 0; i < new_tokens.size() - 1; ++i){
            std::pair<int, int> new_pair(new_tokens[i], new_tokens[i + 1]);
            diff_pairs[new_pair] += wc;
            pair_wordids[new_pair].insert(wordid);

            new_pairs.insert(new_pair);
        }
        
        
    }
}

void updated_affected_word_count(const std::pair<int, int>& merge_pair,
        const std::unordered_set<int>& affected_words, 
        std::unordered_map<int, std::vector<int>> & wordid_encodings, 
        const std::unordered_map<int, long long> & wordid_counts,
        emhash8::HashMap<std::pair<int, int>, int, pair_hash> & pair_counts, 
        std::unordered_map<std::pair<int, int>, std::unordered_set<int>, pair_hash> & pair_wordids, 
        int new_id,
        std::unordered_map<std::pair<int, int>, std::vector<std::vector<int>>, pair_hash> & pair_strings, 
        std::unordered_map<int, std::vector<int>> & vocabulary){
    std::unordered_set<int> affected_words_copy(affected_words);
    
    std::unordered_map<std::pair<int, int>, int, pair_hash> diff_pairs;
    std::unordered_set<std::pair<int, int>, pair_hash> new_pairs;

    fine_grained_pair_counter_diff(affected_words_copy, wordid_encodings, wordid_counts, 
                                        merge_pair, new_id, pair_wordids, 
                                        diff_pairs, new_pairs);

    for(const auto& [pair, count] : diff_pairs){
        if(count == 0){
            continue;
        }
        int& c = (pair_counts[pair] += count);
        if(c <= 0){
             pair_counts.erase(pair);
             pair_wordids.erase(pair);
        }
    }

    for(const auto& new_pair : new_pairs){
        if (pair_strings.find(new_pair) == pair_strings.end()) {
            pair_strings[new_pair] = {vocabulary[new_pair.first], vocabulary[new_pair.second]};
        }        
    }

}

/**
 *     def _count_pairs(word_counts, word_encodings, pair_strings, vocabulary, pair_to_words):
        pair_counts = defaultdict(int)
        for word, count in word_counts.items():
            encoding = word_encodings[word]
            for i in range(0, len(encoding) - 1):
                pair = encoding[i], encoding[i + 1]
                pair_counts[pair] += count
                if pair not in pair_strings:
                    pair_strings[pair] = (vocabulary[pair[0]], vocabulary[pair[1]])

                pair_to_words[pair].add(word)

        return pair_counts
 */

void bpe_train_step2_v2(int vocab_size,          
                std::unordered_map<int, std::vector<int>> & vocabulary, 
                const std::unordered_map<int, long long> & wordid_counts, 
                std::unordered_map<int, std::vector<int>> & wordid_encodings, 
                std::vector<std::pair<std::vector<int>, std::vector<int>>> & merges){
    std::unordered_map<std::pair<int, int>, std::vector<std::vector<int>>, pair_hash> pair_strings;
    emhash8::HashMap<std::pair<int, int>, int, pair_hash>  pair_counts;
    std::unordered_map<std::pair<int, int>, std::unordered_set<int>, pair_hash>  pair_wordids;
    
    std::pair<int, int> pair;
    for(const auto& [wordid, count] : wordid_counts){
        const auto& word_tokens = wordid_encodings[wordid];
        for(int i = 0; i < word_tokens.size() - 1; ++i){
            pair.first = word_tokens[i];
            pair.second = word_tokens[i + 1];
            pair_counts[pair] += count;
            if (pair_strings.find(pair) == pair_strings.end()) {
                pair_strings[pair] = {vocabulary[pair.first], vocabulary[pair.second]};
            }
            pair_wordids[pair].insert(wordid);
        }
    }
    

    bpe_train_step2(vocab_size, pair_counts, pair_strings, vocabulary,
                    pair_wordids, wordid_counts, wordid_encodings, merges);
}

void bpe_train_step2(int vocab_size, 
                emhash8::HashMap<std::pair<int, int>, int, pair_hash> & pair_counts, 
                std::unordered_map<std::pair<int, int>, std::vector<std::vector<int>>, pair_hash> & pair_strings, 
                std::unordered_map<int, std::vector<int>> & vocabulary, 
                std::unordered_map<std::pair<int, int>, std::unordered_set<int>, pair_hash> & pair_wordids, 
                const std::unordered_map<int, long long> & wordid_counts, 
                std::unordered_map<int, std::vector<int>> & wordid_encodings, 
                std::vector<std::pair<std::vector<int>, std::vector<int>>> & merges){
    auto start = std::chrono::steady_clock::now();
    while(vocabulary.size() < vocab_size){
        int max_count = -1;
        std::pair<int, int> max_pair;
        std::vector<std::vector<int>> max_strings;
        for(const auto& [pair, count] : pair_counts){
            if(count > max_count){
                max_count = count;
                max_pair = pair;
                max_strings = pair_strings[pair];
            }else if(count == max_count){
                std::vector<std::vector<int>> strings = pair_strings[pair];
                ComparisonResult r1 = three_way_compare(strings[0], max_strings[0]);
                if(r1 == ComparisonResult::Greater){
                    max_count = count;
                    max_pair = pair;
                    max_strings = strings;
                }else if(r1 == ComparisonResult::Equal){
                    ComparisonResult r2 = three_way_compare(strings[1], max_strings[1]);
                    if(r2 == ComparisonResult::Greater){
                        max_count = count;
                        max_pair = pair;
                        max_strings = strings;                        
                    }
                }
            }
        }

        const std::vector<int>& bytes1 = vocabulary[max_pair.first];
        const std::vector<int>& bytes2 = vocabulary[max_pair.second];
        std::vector<int> merge_bytes;
        merge_bytes.reserve(bytes1.size() + bytes2.size());
        merge_bytes.insert(merge_bytes.end(), bytes1.begin(), bytes1.end());
        merge_bytes.insert(merge_bytes.end(), bytes2.begin(), bytes2.end());

        int size = vocabulary.size();
        vocabulary[size] = merge_bytes;

        auto& affected_words = pair_wordids[max_pair];

        
        updated_affected_word_count(max_pair, affected_words, wordid_encodings, wordid_counts,
                                    pair_counts, pair_wordids, size, pair_strings, vocabulary);
  
        merges.push_back({bytes1, bytes2});


    }
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "bpe_train_step2: " << duration.count() << "ms." << std::endl;
}

void bpe_train_step_test_interface(int vocab_size, 
                emhash8::HashMap<std::pair<int, int>, int, pair_hash> & pair_counts, 
                std::unordered_map<std::pair<int, int>, std::vector<std::vector<int>>, pair_hash> & pair_strings, 
                std::unordered_map<int, std::vector<int>> & vocabulary, 
                std::unordered_map<std::pair<int, int>, std::unordered_set<int>, pair_hash> & pair_wordids, 
                const std::unordered_map<int, long long> & wordid_counts, 
                std::unordered_map<int, std::vector<int>> & wordid_encodings, 
                std::vector<std::pair<std::vector<int>, std::vector<int>>> & merges){
    std::cout << "vocab_size: " << vocab_size << std::endl;
    
    std::cout << "pair_counts: " << pair_counts.size() << std::endl;
    if(pair_counts.size() > 0){
        for(const auto& [key, value] : pair_counts){
            std::cout << "\t[" << key.first << ", " << key.second << "] = " << value << std::endl;
            break;
        }
    }
    std::cout << "pair_strings: " << pair_strings.size() << std::endl;
    if(pair_strings.size() > 0){
        for(const auto& [key, value] : pair_strings){
            std::cout << "\t[" << key.first << ", " << key.second << "] = " << std::endl;
            for(const auto& v : value){
                std::cout << "\t\t";
                for(const auto& vv : v){
                    std::cout << vv;
                }
                std::cout << std::endl;
            }
            break;
        }        
    }
    std::cout << "vocabulary: " << vocabulary.size() << std::endl;
    if(vocabulary.size() > 0){
        for(const auto& [key, value] : vocabulary){
            std::cout <<"\t" << key << std::endl;
            for(const auto& v : value){
                std::cout << "\t\t" << v;
                std::cout << std::endl;
            }
            break;
        }
    }
    std::cout << "pair_wordids: " << pair_wordids.size() << std::endl;
    if(pair_wordids.size() > 0){
        for(const auto& [key, value] : pair_wordids){
            std::cout << "\t[" << key.first << ", " << key.second << "] = " << std::endl;
            for(const auto& v : value){
                std::cout << "\t\t" << v;
                std::cout << std::endl;
            }
            break;   
        }     
    }
    std::cout << "wordid_counts: " << wordid_counts.size() << std::endl;
    if(wordid_counts.size() > 0){
        for(const auto& [key, value] : wordid_counts){
            std::cout <<"\t" << key <<": " << value << std::endl;
            break;   
        }  
    }
    std::cout << "wordid_encodings: " << wordid_encodings.size() << std::endl;
    if(wordid_encodings.size() > 0){
        for(const auto& [key, value] : wordid_encodings){
            std::cout <<"\t" << key << std::endl;
            for(const auto& v : value){
                std::cout << "\t\t" << v;
                std::cout << std::endl;
            }
            break;   
        }           
    }
    std::cout << "merges: " << merges.size() << std::endl;

    {
        std::vector<int> m1 = {1, 2};
        std::vector<int> m2 = {3, 4};
        std::pair<std::vector<int>, std::vector<int>> pair = {m1, m2};
        merges.push_back(pair);
    }

    {
        vocabulary[22222] = {1,3,5};
    }

}