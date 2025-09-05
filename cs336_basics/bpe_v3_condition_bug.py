import regex as re
from collections import defaultdict
import multiprocessing as mp
import time

CHUNK_SIZE = 1024 *  50
N_BYTES = 256
NUM_COUNTER_PROCESS = 32
NUM_MERGER_PROCESS = 1

class BPE_Trainer():
    def train(self, input_path, vocab_size, special_tokens):
        start_time = time.perf_counter()
        word_counts = self._pretokenize_and_count_mp(input_path, special_tokens)
        end_time = time.perf_counter()

        print(f"_pretokenize_and_count_mp: {end_time - start_time}")
        vocabulary = {i: bytes([i]) for i in range(N_BYTES)} # every byte
        for i, token in enumerate(special_tokens):
            vocabulary[N_BYTES + i] = token.encode('utf-8')
        size = N_BYTES + len(special_tokens)
        merges = []

        # initial word encodings are utf-8
        word_encodings = {}
        for word in word_counts:
            word_encodings[word] = list(word.encode('utf-8'))

        pair_strings = {}
        pair_to_words = defaultdict(set)
        pair_counts = BPE_Trainer._count_pairs(word_counts, word_encodings, pair_strings, vocabulary, pair_to_words)

        while size < vocab_size:
            BPE_Trainer._merge_a_pair(pair_counts, pair_strings, vocabulary,
                                   pair_to_words, word_counts, word_encodings,
                                   merges, size)
            size += 1
      
        
        return vocabulary, merges

    @staticmethod
    def _merge_a_pair(pair_counts, pair_strings, vocabulary, pair_to_words, 
                   word_counts, word_encodings, merges, size):
        merge_pair, max_count = max(pair_counts.items(), key = lambda x: (x[1], pair_strings[x[0]]))
        merge_bytes = vocabulary[merge_pair[0]] + vocabulary[merge_pair[1]]

        vocabulary[size] = merge_bytes
        new_id = size


        affected_words = pair_to_words[merge_pair]
        
        # update affected words' counts
        BPE_Trainer._updated_affected_word_count(merge_pair, affected_words, word_encodings,
                                                    word_counts, pair_counts,
                                                    pair_to_words, new_id, pair_strings, vocabulary)

        merges.append((vocabulary[merge_pair[0]], vocabulary[merge_pair[1]]))


    @staticmethod
    def _updated_affected_word_count(merge_pair, affected_words, word_encodings, 
                                     word_counts, pair_counts, pair_to_words, 
                                     new_id, pair_strings, vocabulary):
            # we may update/delete words when iterate it.
            affected_words = affected_words.copy()

            for word in affected_words:
                word_tokens = word_encodings[word]
                wc = word_counts[word]

                for i in range(len(word_tokens) - 1):
                    old_pair = (word_tokens[i], word_tokens[i + 1])
                    pair_counts[old_pair] -= wc
                    if pair_counts[old_pair] <= 0:
                        # we accounted for all occurrences of this pair
                        del pair_counts[old_pair]
                        pair_to_words.pop(old_pair)
                    else:
                        pair_to_words[old_pair].discard(word)


                i = 0
                new_tokens = []                
 
                while i < len(word_tokens):
                    if i < len(word_tokens) - 1 and (word_tokens[i], word_tokens[i + 1]) == merge_pair:
                        new_tokens.append(new_id)
                        i += 2
                    else:
                        new_tokens.append(word_tokens[i])
                        i += 1

                word_encodings[word] = new_tokens

                for i in range(len(new_tokens) - 1):
                    new_pair = (new_tokens[i], new_tokens[i + 1])
                    
                    pair_counts[new_pair] += wc
                    pair_to_words[new_pair].add(word)
                    if new_pair not in pair_strings:
                        pair_strings[new_pair] = (vocabulary[new_pair[0]], vocabulary[new_pair[1]])

    @staticmethod    
    def _count_pairs(word_counts, word_encodings, pair_strings, vocabulary, pair_to_words):
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
    

    @staticmethod
    def _chunk_documents_streaming(
        path: str,
        chunk_size: int = CHUNK_SIZE,
        special_token: str = "<|endoftext|>"
    ):
        """
        Reads 'path' in streaming fashion, yielding chunks of text that
        each end on a '<|endoftext|>' boundary.
        """

        leftover = ""
        token_len = len(special_token)

        with open(path, "r", encoding="utf-8") as f:
            while True:
                # read one chunk_size block of text
                block = f.read(chunk_size)
                if not block:
                    # no more data in file
                    break

                # combine leftover from previous iteration + new block
                block = leftover + block
                leftover = ""

                # find the *last* occurrence of the special token in 'block'
                last_eot_idx = block.rfind(special_token)

                if last_eot_idx == -1:
                    # no complete document in this chunk
                    # keep everything in leftover for the next read
                    leftover = block
                else:
                    # up through last_eot_idx is a complete set of docs
                    yield block[: last_eot_idx + token_len]
                    # keep everything after that boundary as leftover
                    leftover = block[last_eot_idx + token_len:]

        # yield leftover text
        if leftover:
            yield leftover

    @staticmethod
    def _chunk_counter_process(chunk_queue, counter_queue, 
                               pattern, special_token_pattern,
                               num_mergers, running_counters, completed_condition):
        process_name = mp.current_process().name

        queue_get_time = 0
        process_chunk_time = 0
        queue_put_time = 0
        while True:
            start_time = time.perf_counter()
            chunk = chunk_queue.get()
            end_time = time.perf_counter()
            queue_get_time += (end_time - start_time)
            if chunk == None:
                break
            start_time = time.perf_counter()
            blocks = re.split(special_token_pattern, chunk)
            counter = defaultdict(int)
            for block in blocks:
                for match in re.finditer(pattern, block):
                    text = match.group(0)
                    counter[text] += 1
            end_time = time.perf_counter()
            process_chunk_time += (end_time - start_time)
            start_time = end_time
            debug_data = (process_name, counter)
            counter_queue.put(debug_data)
            end_time = time.perf_counter()
            queue_put_time += (end_time - start_time)

        #print(f"queue_get_time: {queue_get_time}"
        #      f", process_chunk_time: {process_chunk_time}"
        #      f", queue_put_time: {queue_put_time}")
        print(f"finish: {process_name}")
        start_time = time.perf_counter()
        with completed_condition:
            running_counters.value -= 1
            if running_counters.value == 0: # last one, put sentinel Nones
                print(f"{process_name} is the last, put {num_mergers} None")
                for _ in range(num_mergers):
                    counter_queue.put(None)
                completed_condition.notify_all()
            else:
                print(f"{process_name} is not the last {running_counters.value}")
                completed_condition.wait()
        end_time = time.perf_counter()
        #print(f"lock time: {end_time - start_time}")
        

    @staticmethod
    def _merge_counter_process(counter_queue, merged_queue):
        merged_counter = defaultdict(int)

        while True:
            data = counter_queue.get()
            if data == None:        
                break
            counter = data[1]
            for k,v in counter.items():
                merged_counter[k] += v
        
        print("prepared stop _merge_counter_process")
        merged_queue.put(merged_counter)
        print("stop _merge_counter_process")

    @staticmethod
    def _queue_moniter_process(chunk_queue, counter_queue, merged_queue, event):
        while not event.is_set():
            print(f"chunk_queue: {chunk_queue.qsize()}, counter_queue: {counter_queue.qsize()}, merged_queue: {merged_queue.qsize()}")
            time.sleep(10)

    def _pretokenize_and_count_mp(self, input_path: str, special_tokens: list[str]):
        # pre-compile regex
        pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        # build split pattern
        special_token_pattern = "|".join(re.escape(token) for token in special_tokens)

        chunk_queue = mp.Queue(maxsize=1_000_000)
        counter_queue = mp.Queue(maxsize=1_000_000)
        merged_queue = mp.Queue(maxsize=NUM_MERGER_PROCESS)
        counter_processes = []

        running_counters = mp.Value('i', NUM_COUNTER_PROCESS)
        completed_condition = mp.Condition()    
           
        for i in range(NUM_COUNTER_PROCESS):
            p = mp.Process(target=BPE_Trainer._chunk_counter_process, 
                        args=(chunk_queue, counter_queue, 
                              pattern, special_token_pattern,
                              NUM_MERGER_PROCESS,
                              running_counters, completed_condition),
                        name=f"counter_process-{i+1}")
            p.start()
            counter_processes.append(p)

        merge_processes = []
        for i in range(NUM_MERGER_PROCESS):
            p = mp.Process(target=BPE_Trainer._merge_counter_process, 
                        args=(counter_queue, merged_queue),
                        name=f"merge_process-{i+1}")
            p.start()
            merge_processes.append(p)        

        stop_event = mp.Event()

        #stop_event.set()
        monitor_process = mp.Process(target=BPE_Trainer._queue_moniter_process, 
                                  args=(chunk_queue, counter_queue, merged_queue, stop_event))
        monitor_process.start()

        for chunk in BPE_Trainer._chunk_documents_streaming(input_path):
            chunk_queue.put(chunk)

        for i in range(NUM_COUNTER_PROCESS):
            chunk_queue.put(None)


        # use main process to merge into final counter
        start_time = time.perf_counter()
        if NUM_MERGER_PROCESS == 1:
            word_counts = merged_queue.get()
        else:
            word_counts = merged_queue.get()
            for _ in range(NUM_MERGER_PROCESS - 1):
                counter = merged_queue.get()
                for k,v in counter.items():
                    word_counts[k] += v
        end_time = time.perf_counter()
        print(f"final merge time: {end_time - start_time}")
        print(f"chunk_queue: {chunk_queue.qsize()}, counter_queue: {counter_queue.qsize()}, merged_queue: {merged_queue.qsize()}")
        if counter_queue.qsize() > 0:
            print("bug")
            data = counter_queue.get()
            print(data[0])
            print(data[1])
        # stop moniter and join all processes
        stop_event.set()
        print("join monitor_process") 
        monitor_process.join() 
        for i in range(NUM_COUNTER_PROCESS):
            print(f"start join {i}")
            counter_processes[i].join()
            print(f"finish join {i}")
        #[p.join() for p in counter_processes]
        print("join merge_processes")
        [p.join() for p in merge_processes]
  

        return word_counts