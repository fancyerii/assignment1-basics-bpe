# distutils: language = c++
from cs336_basics import maxheap_heapq as maxheap
from collections import defaultdict

cdef inline int max_int(int a, int b):
    return a if a > b else b 

cdef inline int min_int(int a, int b):
    return a if a < b else b



cpdef void fine_grained_pair_counter_diff_v3(set affected_words, 
                                          word_encodings, 
                                          word_counts, 
                                          tuple merge_pair, 
                                          diff_pairs, 
                                          int new_id, 
                                          pair_to_words):
    cdef str word
    cdef int wc
    cdef int idx
    cdef int first_idx
    cdef int last_idx
    cdef int i
    cdef int tk_len

    for word in affected_words:
        word_tokens = word_encodings[word]
        wc = word_counts[word]

        # find first and last pairs
        idx = 0
        unaffected_pairs = set()
        tk_len = len(word_tokens)
        #first_idx = -1
        while idx < tk_len - 1:
            if word_tokens[idx] == merge_pair[0] and word_tokens[idx+1] == merge_pair[1]:
                first_idx = idx
                break
            idx += 1

        # assert first_idx exists

        idx = tk_len - 2
        while idx > first_idx + 1:
            if word_tokens[idx] == merge_pair[0] and word_tokens[idx+1] == merge_pair[1]:
                last_idx = idx
                break
            idx -= 1
        else:
            last_idx = first_idx

        start_idx = max_int(0, first_idx - 1) # inclusive
        end_idx = min_int(last_idx + 3, tk_len) # exclusive

        # unaffected [0, start_idx)


        for i in range(start_idx):
            pair = word_tokens[i], word_tokens[i + 1]
            unaffected_pairs.add(pair)
        # unaffected [end_idx-1, :-1]
        for i in range(end_idx - 1, tk_len - 1):
            pair = word_tokens[i], word_tokens[i + 1]
            unaffected_pairs.add(pair)                

        for i in range(start_idx, end_idx - 1):
            old_pair = (word_tokens[i], word_tokens[i + 1])
            diff_pairs[old_pair] -= wc
            if old_pair not in unaffected_pairs:
                pair_to_words[old_pair].discard(word)

        new_tokens = []
        all_new_tokens = []
        for i in range(start_idx):
            all_new_tokens.append(word_tokens[i])
        

        i = start_idx

        while i < end_idx:
            if i < end_idx - 1 and (word_tokens[i], word_tokens[i + 1]) == merge_pair:
                new_tokens.append(new_id)
                all_new_tokens.append(new_id)
                i += 2
            else:
                new_tokens.append(word_tokens[i])
                all_new_tokens.append(word_tokens[i])
                i += 1


        for i in range(end_idx, len(word_tokens)):
            all_new_tokens.append(word_tokens[i])
        
        word_encodings[word] = all_new_tokens

        # add new pairs from the updated word
        for i in range(len(new_tokens) - 1):
            new_pair = (new_tokens[i], new_tokens[i + 1])

            diff_pairs[new_pair] += wc
            pair_to_words[new_pair].add(word)


cpdef void _updated_affected_word_count(merge_pair, 
                                        affected_words, 
                                        word_encodings, 
                                        word_counts, 
                                        pair_counts, 
                                        pair_to_words, 
                                        int new_id, 
                                        pair_strings, 
                                        vocabulary, 
                                        pair_heap):
    # we may update/delete words when iterate it.
    affected_words = affected_words.copy()
    diff_pairs = defaultdict(int)

    fine_grained_pair_counter_diff_v3(affected_words, word_encodings, word_counts, merge_pair, diff_pairs, 
                            new_id, pair_to_words)

    for pair, count in diff_pairs.items():
        if count == 0: continue
        pair_counts[pair] += count
        if count > 0: # new pair
            pair_strings[pair] = (vocabulary[pair[0]], vocabulary[pair[1]])
            maxheap.heappush(pair_heap, (pair_counts[pair], pair_strings[pair], pair))
        
        if pair_counts[pair] <= 0: # should not less than 0!
            del pair_counts[pair]
            pair_to_words.pop(pair, None)

cpdef void _merge_a_pair(pair_counts, 
                         pair_strings, 
                         vocabulary, 
                         pair_to_words, 
                         word_counts, 
                         word_encodings, 
                         merges, 
                         int size, 
                         pair_heap):
    
    while pair_heap:
        count, string_priority, merge_pair = maxheap.heappop(pair_heap)
        
        # check pair validity
        if merge_pair in pair_counts and pair_counts[merge_pair] == count:
            break
        elif merge_pair in pair_counts:
            # update count (lazily)
            maxheap.heappush(pair_heap, (pair_counts[merge_pair], 
                                            string_priority, 
                                            merge_pair))


    merge_bytes = vocabulary[merge_pair[0]] + vocabulary[merge_pair[1]]

    vocabulary[size] = merge_bytes
    new_id = size


    affected_words = pair_to_words[merge_pair]
    
    # update affected words' counts
    _updated_affected_word_count(merge_pair, affected_words, word_encodings,
                                                word_counts, pair_counts,
                                                pair_to_words, new_id, pair_strings, 
                                                vocabulary, pair_heap)

    merges.append((vocabulary[merge_pair[0]], vocabulary[merge_pair[1]]))


cpdef void bpe_train_step2(int vocab_size,
                      pair_counts,
                      pair_strings,
                      vocabulary,
                      pair_to_words,
                      word_counts,
                      word_encodings,
                      merges,
                      pair_heap):

    cdef int size = len(vocabulary)
    while size < vocab_size:
        _merge_a_pair(pair_counts, pair_strings, vocabulary,
                                pair_to_words, word_counts, word_encodings,
                                merges, size, pair_heap)
        size += 1
