import regex as re
from collections import defaultdict
import multiprocessing as mp
import time

CHUNK_SIZE = 1024 *  50
N_BYTES = 256
NUM_COUNTER_PROCESS = 8
NUM_MERGER_PROCESS = 1

class BPE_Trainer():
    def dump_step1_results(self, input_path, vocab_size, special_tokens, 
                    output_dir,
                    num_counter=NUM_COUNTER_PROCESS,
                    num_merger=NUM_MERGER_PROCESS,
                    do_monitor=False,
                    **unused):
        start_time = time.perf_counter()
        word_counts = self._pretokenize_and_count_mp(input_path, special_tokens, num_counter, num_merger, do_monitor)
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
        print(f"dump to {output_dir}")
        import os
        import json
        os.makedirs(output_dir, exist_ok=True)

        start_time = time.perf_counter()
        word_ids = {word:id for id, word in enumerate(word_counts)}
        end_time = time.perf_counter()
        print(f"word_ids: {end_time - start_time:.2f}s len: {len(word_ids)}")
        start_time = time.perf_counter()
        with open(os.path.join(output_dir, "word_ids.json"), 'w') as json_file:
            json.dump(word_ids, json_file, indent=4)
        end_time = time.perf_counter()
        print(f"dump word_ids: {end_time - start_time:.2f}s")

        start_time = time.perf_counter()
        wordid_counts = {word_ids[word]:count for word, count in word_counts.items()}
        end_time = time.perf_counter()
        print(f"wordid_counts: {end_time - start_time:.2f}s len: {len(wordid_counts)}")
        start_time = time.perf_counter()
        with open(os.path.join(output_dir, "wordid_counts.json"), 'w') as json_file:
            json.dump(wordid_counts, json_file, indent=4)  
        end_time = time.perf_counter()
        print(f"dump wordid_counts: {end_time - start_time:.2f}s")

        start_time = time.perf_counter()
        wordid_encodings = {word_ids[word]:encoding for word, encoding in word_encodings.items()}      
        end_time = time.perf_counter()
        print(f"wordid_encodings: {end_time - start_time:.2f}s len: {len(wordid_encodings)}")
        start_time = time.perf_counter()
        with open(os.path.join(output_dir, "wordid_encodings.json"), 'w') as json_file:
            json.dump(wordid_encodings, json_file, indent=4) 
        end_time = time.perf_counter()
        print(f"dump wordid_encodings: {end_time - start_time:.2f}s")

        start_time = time.perf_counter()
        pair_strings_json = {}
        for pair, string in pair_strings.items():
            key = ",".join(str(item) for item in pair)
            value = [list(item) for item in string]
            pair_strings_json[key] = value
        end_time = time.perf_counter()
        print(f"pair_strings_json: {end_time - start_time:.2f}s len: {len(pair_strings_json)}")
        start_time = time.perf_counter()
        with open(os.path.join(output_dir, "pair_strings.json"), 'w') as json_file:
            json.dump(pair_strings_json, json_file, indent=4)   
        end_time = time.perf_counter()
        print(f"dump pair_strings_json: {end_time - start_time:.2f}s")

        start_time = time.perf_counter()
        vocabulary_json = {key:list(value) for key, value in vocabulary.items()}      
        end_time = time.perf_counter()
        print(f"vocabulary_json: {end_time - start_time:.2f}s len: {len(vocabulary_json)}")
        start_time = time.perf_counter()
        with open(os.path.join(output_dir, "vocabulary.json"), 'w') as json_file:
            json.dump(vocabulary_json, json_file, indent=4)    
        end_time = time.perf_counter()
        print(f"dump vocabulary_json: {end_time - start_time:.2f}s")

        start_time = time.perf_counter()
        pair_to_wordids = {}
        for pair, words in pair_to_words.items():
            key = ",".join(str(item) for item in pair)
            wordids = [word_ids[word] for word in words]
            pair_to_wordids[key] = wordids
        end_time = time.perf_counter()
        print(f"pair_to_wordids: {end_time - start_time:.2f}s len: {len(pair_to_wordids)}")
        start_time = time.perf_counter()
        with open(os.path.join(output_dir, "pair_to_wordids.json"), 'w') as json_file:
            json.dump(pair_to_wordids, json_file, indent=4) 
        end_time = time.perf_counter()
        print(f"dump pair_to_wordids: {end_time - start_time:.2f}s")

        start_time = time.perf_counter()
        pair_counts_json = {}
        for pair, count in pair_counts.items():
            key = ",".join(str(item) for item in pair)
            pair_counts_json[key] = count
        end_time = time.perf_counter()
        print(f"pair_counts_json: {end_time - start_time:.2f}s len: {len(pair_counts_json)}")
        start_time = time.perf_counter()            
        with open(os.path.join(output_dir, "pair_counts.json"), 'w') as json_file:
            json.dump(pair_counts_json, json_file, indent=4)         
        end_time = time.perf_counter()
        print(f"dump pair_counts_json: {end_time - start_time:.2f}s") 


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
                               pattern, special_token_pattern):
        while True:
            chunk = chunk_queue.get()
            if chunk == None:
                break
            blocks = re.split(special_token_pattern, chunk)
            counter = defaultdict(int)
            for block in blocks:
                for match in re.finditer(pattern, block):
                    text = match.group(0)
                    counter[text] += 1
            counter_queue.put(counter)
                 

    @staticmethod
    def _merge_counter_process(counter_queue, merged_queue):
        merged_counter = defaultdict(int)

        while True:
            counter = counter_queue.get()
            if counter == None:        
                break

            for k,v in counter.items():
                merged_counter[k] += v

        merged_queue.put(merged_counter)

    @staticmethod
    def _queue_moniter_process(chunk_queue, counter_queue, merged_queue, event):
        while not event.is_set():
            print(f"chunk_queue: {chunk_queue.qsize()}, counter_queue: {counter_queue.qsize()}, merged_queue: {merged_queue.qsize()}")
            time.sleep(10)

    def _pretokenize_and_count_mp(self, input_path: str, special_tokens: list[str],
                                  num_counter, num_merger, do_monitor):
        # pre-compile regex
        pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        # build split pattern
        special_token_pattern = "|".join(re.escape(token) for token in special_tokens)

        chunk_queue = mp.Queue(maxsize=1_000_000)
        counter_queue = mp.Queue(maxsize=1_000_000)
        merged_queue = mp.Queue(maxsize=num_merger)
        counter_processes = []
     
        for i in range(num_counter):
            p = mp.Process(target=BPE_Trainer._chunk_counter_process, 
                        args=(chunk_queue, counter_queue, 
                              pattern, special_token_pattern),
                        name=f"counter_process-{i+1}")
            p.start()
            counter_processes.append(p)

        merge_processes = []
        for i in range(num_merger):
            p = mp.Process(target=BPE_Trainer._merge_counter_process, 
                        args=(counter_queue, merged_queue),
                        name=f"merge_process-{i+1}")
            p.start()
            merge_processes.append(p)        

        

        # stop_event.set() for unit test, we should stop monitor to pass speed test
        # because monitor process will sleep 30s 
        if do_monitor:
            stop_event = mp.Event()

            monitor_process = mp.Process(target=BPE_Trainer._queue_moniter_process, 
                                  args=(chunk_queue, counter_queue, merged_queue, stop_event))
            monitor_process.start()



        for chunk in BPE_Trainer._chunk_documents_streaming(input_path):
            chunk_queue.put(chunk)

        for i in range(num_counter):
            chunk_queue.put(None)

        for p in counter_processes:
            p.join()


        for _ in range(num_merger):
            counter_queue.put(None)



        # use main process to merge into final counter
        if num_merger == 1:
            word_counts = merged_queue.get()
        else:
            word_counts = merged_queue.get()
            for _ in range(num_merger - 1):
                counter = merged_queue.get()
                for k,v in counter.items():
                    word_counts[k] += v

        # stop moniter and join all processes
        
        for p in merge_processes:
            p.join() 

        if do_monitor:
            stop_event.set()
            monitor_process.join()   

        return word_counts