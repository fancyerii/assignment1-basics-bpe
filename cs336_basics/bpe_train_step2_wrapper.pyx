# distutils: language = c++

# 从 .pxd 文件导入声明
from cs336_basics.bpe_train_step2_wrapper cimport bpe_train_step2
from cs336_basics.bpe_train_step2_wrapper cimport bpe_train_step2_v2
from cs336_basics.bpe_train_step2_wrapper cimport bpe_train_step2_v3
from cs336_basics.bpe_train_step2_wrapper cimport bpe_train_step2_v4
from cs336_basics.bpe_train_step2_wrapper cimport bpe_train_step2_v5
from cs336_basics.bpe_train_step2_wrapper cimport bpe_train_step2_v6
from cs336_basics.bpe_train_step2_wrapper cimport pair_hash, HashMap
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libcpp.unordered_set cimport unordered_set
from libcpp.utility cimport pair
import time
import psutil
import os
import gc
import sys


cpdef py_bpe_train_step2_list(input_list):

    current_process = psutil.Process(os.getpid())

    mem_info = current_process.memory_info()
    print("before construct c++")
    print(f"\tCurrent Memory Usage (RSS): {mem_info.rss / 1024 / 1024:.2f} MB")
    print(f"\tCurrent Memory Usage (VMS): {mem_info.vms / 1024 / 1024:.2f} MB")

    vocab_size = input_list[0]
    pair_counts_py = input_list[1]
    pair_strings_py = input_list[2]
    vocabulary_py = input_list[3]
    pair_wordids_py = input_list[4]
    wordid_counts_py = input_list[5]
    wordid_encodings_py = input_list[6]
    merges_py = input_list[7]

    input_list.clear()

    # 声明 C++ 容器
    cdef HashMap[pair[int, int], int, pair_hash] pair_counts_cpp
    cdef unordered_map[pair[int, int], vector[vector[int]], pair_hash] pair_strings_cpp
    cdef unordered_map[int, vector[int]] vocabulary_cpp
    cdef unordered_map[pair[int, int], unordered_set[int], pair_hash] pair_wordids_cpp
    cdef unordered_map[int, long long] wordid_counts_cpp
    cdef unordered_map[int, vector[int]] wordid_encodings_cpp
    cdef vector[pair[vector[int], vector[int]]] merges_cpp

 
    cdef pair[int, int] pair_key
    cdef vector[vector[int]] strings_value
    cdef vector[int] vector_value
    cdef unordered_set[int] set_value




    for p, count in pair_counts_py.items():
        pair_key.first = p[0]
        pair_key.second = p[1]
        pair_counts_cpp[pair_key] = count

    for p, string in pair_strings_py.items():
        pair_key.first = p[0]
        pair_key.second = p[1]
        strings_value.clear()
        value = [list(item) for item in string] 
        vector_value = value[0]
        strings_value.push_back(vector_value)
        vector_value = value[1]
        strings_value.push_back(vector_value)        
        pair_strings_cpp[pair_key] = strings_value     
    
    for k, v in vocabulary_py.items():
        value = list(v)
        vector_value = value
        vocabulary_cpp[k] = vector_value

    for p, wordids in pair_wordids_py.items():
        pair_key.first = p[0]
        pair_key.second = p[1]        
        set_value = wordids
        pair_wordids_cpp[pair_key] = set_value

    for k, v in wordid_counts_py.items():
        wordid_counts_cpp[k] = v

    for k, v in wordid_encodings_py.items():
        vector_value = v
        wordid_encodings_cpp[k] = vector_value

    mem_info = current_process.memory_info()
    print("after construct c++")
    print(f"\tCurrent Memory Usage (RSS): {mem_info.rss / 1024 / 1024:.2f} MB")
    print(f"\tCurrent Memory Usage (VMS): {mem_info.vms / 1024 / 1024:.2f} MB")

    # 删除python变量
    print(f"pair_counts_py: {sys.getrefcount(pair_counts_py)}")
    print(f"pair_strings_py: {sys.getrefcount(pair_strings_py)}")
    print(f"vocabulary_py: {sys.getrefcount(vocabulary_py)}")
    print(f"pair_wordids_py: {sys.getrefcount(pair_wordids_py)}")
    print(f"wordid_counts_py: {sys.getrefcount(wordid_counts_py)}")
    print(f"wordid_encodings_py: {sys.getrefcount(wordid_encodings_py)}")
    print(f"merges_py: {sys.getrefcount(merges_py)}")

    pair_counts_py = None
    pair_strings_py = None
    vocabulary_py = None
    pair_wordids_py = None
    wordid_counts_py = None
    wordid_encodings_py = None
    merges_py = None


    

    gc.collect()
    mem_info = current_process.memory_info()
    print("after gc")
    print(f"\tCurrent Memory Usage (RSS): {mem_info.rss / 1024 / 1024:.2f} MB")
    print(f"\tCurrent Memory Usage (VMS): {mem_info.vms / 1024 / 1024:.2f} MB")

    start_time = time.time()
    bpe_train_step2(vocab_size,
                    pair_counts_cpp,
                    pair_strings_cpp,
                    vocabulary_cpp,
                    pair_wordids_cpp,
                    wordid_counts_cpp,
                    wordid_encodings_cpp,
                    merges_cpp)
    end_time = time.time()
    print(f"py_bpe_train_step2_list in wrapper: {end_time - start_time:.2f}s")

    return merges_cpp, vocabulary_cpp    

cpdef py_bpe_train_step2(int vocab_size,
                             pair_counts_py,
                             pair_strings_py,
                             vocabulary_py,
                             pair_wordids_py,
                             wordid_counts_py,
                             wordid_encodings_py,
                             merges_py):

    # 声明 C++ 容器
    cdef HashMap[pair[int, int], int, pair_hash] pair_counts_cpp
    cdef unordered_map[pair[int, int], vector[vector[int]], pair_hash] pair_strings_cpp
    cdef unordered_map[int, vector[int]] vocabulary_cpp
    cdef unordered_map[pair[int, int], unordered_set[int], pair_hash] pair_wordids_cpp
    cdef unordered_map[int, long long] wordid_counts_cpp
    cdef unordered_map[int, vector[int]] wordid_encodings_cpp
    cdef vector[pair[vector[int], vector[int]]] merges_cpp

 
    cdef pair[int, int] pair_key
    cdef vector[vector[int]] strings_value
    cdef vector[int] vector_value
    cdef unordered_set[int] set_value




    for p, count in pair_counts_py.items():
        pair_key.first = p[0]
        pair_key.second = p[1]
        pair_counts_cpp[pair_key] = count

    for p, string in pair_strings_py.items():
        pair_key.first = p[0]
        pair_key.second = p[1]
        strings_value.clear()
        value = [list(item) for item in string] 
        vector_value = value[0]
        strings_value.push_back(vector_value)
        vector_value = value[1]
        strings_value.push_back(vector_value)        
        pair_strings_cpp[pair_key] = strings_value     
    
    for k, v in vocabulary_py.items():
        value = list(v)
        vector_value = value
        vocabulary_cpp[k] = vector_value

    for p, wordids in pair_wordids_py.items():
        pair_key.first = p[0]
        pair_key.second = p[1]        
        set_value = wordids
        pair_wordids_cpp[pair_key] = set_value

    for k, v in wordid_counts_py.items():
        wordid_counts_cpp[k] = v

    for k, v in wordid_encodings_py.items():
        vector_value = v
        wordid_encodings_cpp[k] = vector_value

    # 调用 C++ 函数
    bpe_train_step2(vocab_size,
                    pair_counts_cpp,
                    pair_strings_cpp,
                    vocabulary_cpp,
                    pair_wordids_cpp,
                    wordid_counts_cpp,
                    wordid_encodings_cpp,
                    merges_cpp)

    return merges_cpp, vocabulary_cpp


cpdef py_bpe_train_step2_v2(int vocab_size,
                             vocabulary_py,
                             wordid_counts_py,
                             wordid_encodings_py,
                             merges_py):

    # 声明 C++ 容器
    cdef unordered_map[int, vector[int]] vocabulary_cpp
    cdef unordered_map[int, long long] wordid_counts_cpp
    cdef unordered_map[int, vector[int]] wordid_encodings_cpp
    cdef vector[pair[vector[int], vector[int]]] merges_cpp

 
    cdef pair[int, int] pair_key
    cdef vector[vector[int]] strings_value
    cdef vector[int] vector_value
    cdef unordered_set[int] set_value
 
    
    for k, v in vocabulary_py.items():
        value = list(v)
        vector_value = value
        vocabulary_cpp[k] = vector_value


    for k, v in wordid_counts_py.items():
        wordid_counts_cpp[k] = v

    for k, v in wordid_encodings_py.items():
        vector_value = v
        wordid_encodings_cpp[k] = vector_value

    # 调用 C++ 函数
    bpe_train_step2_v2(vocab_size,
                    vocabulary_cpp,
                    wordid_counts_cpp,
                    wordid_encodings_cpp,
                    merges_cpp)

    return merges_cpp, vocabulary_cpp


cpdef py_bpe_train_step2_opt(int vocab_size,
                             vocabulary_py,
                             wordid_counts_py,
                             wordid_encodings_py,
                             merges_py):

    # 声明 C++ 容器
    cdef unordered_map[int, vector[int]] vocabulary_cpp
    cdef unordered_map[int, long long] wordid_counts_cpp
    cdef unordered_map[int, vector[int]] wordid_encodings_cpp
    cdef vector[pair[vector[int], vector[int]]] merges_cpp

 
    vocabulary_cpp = vocabulary_py

    wordid_counts_cpp = wordid_counts_py

    wordid_encodings_cpp = wordid_encodings_py
    # 调用 C++ 函数
    bpe_train_step2_v2(vocab_size,
                    vocabulary_cpp,
                    wordid_counts_cpp,
                    wordid_encodings_cpp,
                    merges_cpp)

    return merges_cpp, vocabulary_cpp

cpdef py_bpe_train_step2_v3(int vocab_size,
                             vocabulary_py,
                             wordid_counts_py,
                             wordid_encodings_py,
                             merges_py):

    # 声明 C++ 容器
    cdef unordered_map[int, vector[int]] vocabulary_cpp
    cdef unordered_map[int, long long] wordid_counts_cpp
    cdef unordered_map[int, vector[int]] wordid_encodings_cpp
    cdef vector[pair[vector[int], vector[int]]] merges_cpp

 
    vocabulary_cpp = vocabulary_py

    wordid_counts_cpp = wordid_counts_py

    wordid_encodings_cpp = wordid_encodings_py
    # 调用 C++ 函数
    bpe_train_step2_v3(vocab_size,
                    vocabulary_cpp,
                    wordid_counts_cpp,
                    wordid_encodings_cpp,
                    merges_cpp)

    return merges_cpp, vocabulary_cpp



cpdef py_bpe_train_step2_v4(int vocab_size,
                             vocabulary_py,
                             wordid_counts_py,
                             wordid_encodings_py,
                             merges_py):

    # 声明 C++ 容器
    cdef unordered_map[int, vector[int]] vocabulary_cpp
    cdef unordered_map[int, long long] wordid_counts_cpp
    cdef unordered_map[int, vector[int]] wordid_encodings_cpp
    cdef vector[pair[vector[int], vector[int]]] merges_cpp

 
    vocabulary_cpp = vocabulary_py

    wordid_counts_cpp = wordid_counts_py

    wordid_encodings_cpp = wordid_encodings_py
    # 调用 C++ 函数
    bpe_train_step2_v4(vocab_size,
                    vocabulary_cpp,
                    wordid_counts_cpp,
                    wordid_encodings_cpp,
                    merges_cpp)

    return merges_cpp, vocabulary_cpp


cpdef py_bpe_train_step2_v5(int vocab_size,
                             vocabulary_py,
                             wordid_counts_py,
                             wordid_encodings_py,
                             merges_py):

    # 声明 C++ 容器
    cdef unordered_map[int, vector[int]] vocabulary_cpp
    cdef unordered_map[int, long long] wordid_counts_cpp
    cdef unordered_map[int, vector[int]] wordid_encodings_cpp
    cdef vector[pair[vector[int], vector[int]]] merges_cpp

 
    vocabulary_cpp = vocabulary_py

    wordid_counts_cpp = wordid_counts_py

    wordid_encodings_cpp = wordid_encodings_py
    # 调用 C++ 函数
    bpe_train_step2_v5(vocab_size,
                    vocabulary_cpp,
                    wordid_counts_cpp,
                    wordid_encodings_cpp,
                    merges_cpp)

    return merges_cpp, vocabulary_cpp

cpdef py_bpe_train_step2_v6(int vocab_size,
                             vocabulary_py,
                             wordid_counts_py,
                             wordid_encodings_py,
                             merges_py):

    # 声明 C++ 容器
    cdef unordered_map[int, vector[int]] vocabulary_cpp
    cdef unordered_map[int, long long] wordid_counts_cpp
    cdef unordered_map[int, vector[int]] wordid_encodings_cpp
    cdef vector[pair[vector[int], vector[int]]] merges_cpp

 
    vocabulary_cpp = vocabulary_py

    wordid_counts_cpp = wordid_counts_py

    wordid_encodings_cpp = wordid_encodings_py
    # 调用 C++ 函数
    bpe_train_step2_v6(vocab_size,
                    vocabulary_cpp,
                    wordid_counts_cpp,
                    wordid_encodings_cpp,
                    merges_cpp)

    return merges_cpp, vocabulary_cpp