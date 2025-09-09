import regex as re
from collections import defaultdict
import multiprocessing as mp
import time
import argparse

CHUNK_SIZE = 1024 * 1024
N_BYTES = 256
NUM_COUNTER_PROCESS = 8
NUM_MERGER_PROCESS = 1

class BPE_Trainer():
    def train(self, input_path, vocab_size, special_tokens, *args):
        parser = argparse.ArgumentParser()
        parser.add_argument("--num_counter", 
                            "-c",
                            type=int, 
                            default=NUM_COUNTER_PROCESS, 
                            help="number of processes for counting")
        parser.add_argument("--num_merger", 
                            "-m",
                            type=int, 
                            default=NUM_MERGER_PROCESS, 
                            help="number of processes for merging")
        parser.add_argument("--do_monitor",
                            action="store_true",
                            help="Enable queue monitor. (default: False)"
        )        
        parser.add_argument("--skip_merge",
                            action="store_true",
                            help="skip merge (default: False)"
        ) 
        parser.add_argument("--chunk_size", 
                            type=str, 
                            default=f"{CHUNK_SIZE}", 
                            help="chunk_size")
        
        args = parser.parse_args(args)
        print(f"train: {args=}")
        chunk_size = args.chunk_size.lower()
        if chunk_size.endswith("kb"):
            chunk_size = int(chunk_size[:-2].strip()) * 1024
        elif chunk_size.endswith("mb"):
            chunk_size = int(chunk_size[:-2].strip()) * 1024 * 1024
        else:
            chunk_size = int(chunk_size.strip())
        print(f"{chunk_size=}")
        

        num_counter = args.num_counter
        num_merger = args.num_merger
        do_monitor = args.do_monitor

        start_time = time.perf_counter()
        word_counts = self._pretokenize_and_count_mp(input_path, special_tokens, num_counter, 
                                                     num_merger, do_monitor, chunk_size)
        end_time = time.perf_counter()

        print(f"_pretokenize_and_count_mp: {end_time - start_time}")



        vocabulary = {i: bytes([i]) for i in range(N_BYTES)} # every byte
        for i, token in enumerate(special_tokens):
            vocabulary[N_BYTES + i] = token.encode('utf-8')
        size = N_BYTES + len(special_tokens)
        merges = []

        if args.skip_merge:
            print(f"skip merge")
            return vocabulary, merges 

        # initial word encodings are utf-8
        word_encodings = {}
        for word in word_counts:
            word_encodings[word] = list(word.encode('utf-8'))

        pair_strings = {}
        pair_to_words = defaultdict(set)
        pair_counts = BPE_Trainer._count_pairs(word_counts, word_encodings, pair_strings, vocabulary, pair_to_words)

        max_time = [0]
        update_time = [0]
        start_time = time.perf_counter()
        while size < vocab_size:
            BPE_Trainer._merge_a_pair(pair_counts, pair_strings, vocabulary,
                                   pair_to_words, word_counts, word_encodings,
                                   merges, size, max_time, update_time)
            size += 1
        end_time = time.perf_counter()
        print(f"merge time: {end_time - start_time}")        
        print(f"\tmax_time: {max_time[0]}, update_time: {update_time[0]}")
        return vocabulary, merges

    @staticmethod
    def _merge_a_pair(pair_counts, pair_strings, vocabulary, pair_to_words, 
                   word_counts, word_encodings, merges, size,
                   max_time, update_time):
        start_time = time.perf_counter()
        merge_pair, max_count = max(pair_counts.items(), key = lambda x: (x[1], pair_strings[x[0]]))
        end_time = time.perf_counter()
        max_time[0] += (end_time - start_time)

        merge_bytes = vocabulary[merge_pair[0]] + vocabulary[merge_pair[1]]

        vocabulary[size] = merge_bytes
        new_id = size


        affected_words = pair_to_words[merge_pair]
        
        # update affected words' counts
        start_time = time.perf_counter()
        BPE_Trainer._updated_affected_word_count(merge_pair, affected_words, word_encodings,
                                                    word_counts, pair_counts,
                                                    pair_to_words, new_id, pair_strings, vocabulary)
        end_time = time.perf_counter()
        update_time[0] += (end_time - start_time)
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

        leftover = b""
        special_token_bytes = special_token.encode("utf-8")
        token_len = len(special_token)

        with open(path, "rb") as f:
            while True:
                # read one chunk_size block of text
                block = f.read(chunk_size)
                if not block:
                    # no more data in file
                    break

                # combine leftover from previous iteration + new block
                block = leftover + block
                leftover = b""

                # find the *last* occurrence of the special token in 'block'
                last_eot_idx = block.rfind(special_token_bytes)

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
            chunk = chunk.decode("utf-8")
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
                                  num_counter, num_merger, do_monitor, chunk_size):
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



        for chunk in BPE_Trainer._chunk_documents_streaming(input_path, chunk_size):
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