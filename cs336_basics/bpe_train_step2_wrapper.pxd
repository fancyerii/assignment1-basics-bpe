# distutils: language = c++

# 导入 C++ 标准库类型
from libcpp.utility cimport pair
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libcpp.unordered_set cimport unordered_set


cdef extern from "../lib_bpe_train_step2/include/bpe_train_step2.h" :
    cppclass pair_hash:
        pass

cdef extern from "../lib_bpe_train_step2/include/emhash/hash_table8.hpp" namespace "emhash8":
    cppclass HashMap[K, V, H]:
        #ValueT& operator[](const KeyT& key) noexcept
        V& operator[](const K& key)


cdef extern from "../lib_bpe_train_step2/include/bpe_train_step2.h":
    void bpe_train_step2(int vocab_size,
                         HashMap[pair[int, int], int, pair_hash] & pair_counts,
                         unordered_map[pair[int, int], vector[vector[int]], pair_hash] & pair_strings,
                         unordered_map[int, vector[int]] & vocabulary,
                         unordered_map[pair[int, int], unordered_set[int], pair_hash] & pair_wordids,
                         const unordered_map[int, long long] & wordid_counts,
                         unordered_map[int, vector[int]] & wordid_encodings,
                         vector[pair[vector[int], vector[int]]] & merges) except +
    
    void bpe_train_step2_v2(int vocab_size,
                         unordered_map[int, vector[int]] & vocabulary,
                         const unordered_map[int, long long] & wordid_counts,
                         unordered_map[int, vector[int]] & wordid_encodings,
                         vector[pair[vector[int], vector[int]]] & merges) except +