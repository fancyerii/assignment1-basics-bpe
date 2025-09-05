# distutils: language = c++

cdef inline int max_int(int a, int b):
    return a if a > b else b 

cdef inline int min_int(int a, int b):
    return a if a < b else b

cpdef void fine_grained_pair_counter_diff_v2(set affected_words, 
                                          word_encodings, 
                                          word_counts, 
                                          tuple merge_pair, 
                                          diff_pairs, 
                                          int new_id, 
                                          pair_to_words, 
                                          set new_pairs):
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

        # TODO avoid slice copy
        # affected_tokens = word_tokens[start_idx: end_idx]
        # for i in range(len(affected_tokens) - 1):
        #     old_pair = (affected_tokens[i], affected_tokens[i + 1])
        #     diff_pairs[old_pair] -= wc 
        #     if old_pair not in unaffected_pairs:   
        #         pair_to_words[old_pair].discard(word)
        for i in range(start_idx, end_idx - 1):
            old_pair = (word_tokens[i], word_tokens[i + 1])
            diff_pairs[old_pair] -= wc
            if old_pair not in unaffected_pairs:
                pair_to_words[old_pair].discard(word)

        new_tokens = []
        all_new_tokens = []
        for i in range(start_idx):
            all_new_tokens.append(word_tokens[i])
        
        #i = 0
        i = start_idx
        # account for multiple occurrences of the pair
        # while i < len(affected_tokens):
        #     if i < len(affected_tokens) - 1 and (affected_tokens[i], affected_tokens[i + 1]) == merge_pair:
        #         new_tokens.append(new_id)
        #         all_new_tokens.append(new_id)
        #         # jump past pair
        #         i += 2
        #     else:
        #         new_tokens.append(affected_tokens[i])
        #         all_new_tokens.append(affected_tokens[i])
        #         i += 1
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

            new_pairs.add(new_pair)


cpdef void fine_grained_pair_counter_diff(set affected_words, 
                                          word_encodings, 
                                          word_counts, 
                                          tuple merge_pair, 
                                          diff_pairs, 
                                          int new_id, 
                                          pair_to_words, 
                                          set new_pairs):
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

        # TODO avoid slice copy
        affected_tokens = word_tokens[start_idx: end_idx]
        for i in range(len(affected_tokens) - 1):
            old_pair = (affected_tokens[i], affected_tokens[i + 1])
            diff_pairs[old_pair] -= wc 
            if old_pair not in unaffected_pairs:   
                pair_to_words[old_pair].discard(word)
        

        new_tokens = []
        all_new_tokens = []
        for i in range(start_idx):
            all_new_tokens.append(word_tokens[i])
        
        i = 0
        # account for multiple occurrences of the pair
        while i < len(affected_tokens):
            if i < len(affected_tokens) - 1 and (affected_tokens[i], affected_tokens[i + 1]) == merge_pair:
                new_tokens.append(new_id)
                all_new_tokens.append(new_id)
                # jump past pair
                i += 2
            else:
                new_tokens.append(affected_tokens[i])
                all_new_tokens.append(affected_tokens[i])
                i += 1



        for i in range(end_idx, len(word_tokens)):
            all_new_tokens.append(word_tokens[i])
        
        word_encodings[word] = all_new_tokens

        # add new pairs from the updated word
        for i in range(len(new_tokens) - 1):
            new_pair = (new_tokens[i], new_tokens[i + 1])

            diff_pairs[new_pair] += wc
            pair_to_words[new_pair].add(word)

            new_pairs.add(new_pair)