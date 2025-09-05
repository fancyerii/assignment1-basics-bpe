import regex as re
from collections import defaultdict
import multiprocessing as mp
import time
import argparse
from cs336_basics.bpe_train_step2_wrapper import py_bpe_train_step2

CHUNK_SIZE = 1024 *  50
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
        
        args = parser.parse_args(args)
        print(f"train: {args=}")
        num_counter = args.num_counter
        num_merger = args.num_merger
        do_monitor = args.do_monitor

        start_time = time.perf_counter()
        word_counts = self._pretokenize_and_count_mp(input_path, special_tokens, num_counter, num_merger, do_monitor)
        end_time = time.perf_counter()

        print(f"_pretokenize_and_count_mp: {end_time - start_time}")

        start_time = time.time()
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


        word_ids = {word:id for id, word in enumerate(word_counts)}

        wordid_counts = {word_ids[word]:count for word, count in word_counts.items()}

        wordid_encodings = {word_ids[word]:encoding for word, encoding in word_encodings.items()}      

        pair_to_wordids = {pair:[word_ids[word] for word in words] for pair, words in pair_to_words.items()}

        end_time = time.time()
        print(f"prepare & convert: {end_time - start_time:.2f}")
        start_time = time.time()
        merges_cpp, vocabulary_cpp = py_bpe_train_step2(vocab_size, 
                             pair_counts,
                             pair_strings,
                             vocabulary,
                             pair_to_wordids,
                             wordid_counts,
                             wordid_encodings,
                             merges)
        end_time = time.time()
        print(f"py_bpe_train_step2: {end_time - start_time:.2f}")

        vocabulary = {k:bytes(v) for k, v in vocabulary_cpp.items()}
        merges = [(bytes(arr[0]), bytes(arr[1])) for arr in merges_cpp]
        return vocabulary, merges

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
    
if __name__ == '__main__':
    data_path = "./data/TinyStoriesV2-GPT4-valid.txt"
    bpe_trainer = BPE_Trainer()
    vocab_size = 10000
    start_time = time.perf_counter()
    vocabulary, merges = bpe_trainer.train(data_path, vocab_size,["<|endoftext|>"], *("-c 1".split()))
    end_time = time.perf_counter()
    print(f"TinyStories time: {end_time - start_time:.2f} seconds")
    print(merges[:100])    