import regex as re
from collections import defaultdict


CHUNK_SIZE = 1024 *  50
N_BYTES = 256

import time
class BPE_Trainer():
    def train(self, input_path, vocab_size, special_tokens, *args):
        stime = time.perf_counter()
        word_counts = self._pretokenize_and_count(input_path, special_tokens)
        etime = time.perf_counter()
        print(f"_pretokenize_and_count time {etime - stime}")

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
        count_pairs_time = 0
        max_time = 0
        update_time = 0
        while size < vocab_size:
            stime2 = time.perf_counter()
            pair_counts = BPE_Trainer._count_pairs(word_counts, word_encodings, pair_strings, vocabulary)
            etime2 = time.perf_counter()
            count_pairs_time += (etime2 - stime2)
            stime2 = etime2
            merge_pair, max_count = max(pair_counts.items(), key = lambda x: (x[1], pair_strings[x[0]]))
            etime2 = time.perf_counter()
            max_time += (etime2 - stime2)
            merge_bytes = vocabulary[merge_pair[0]] + vocabulary[merge_pair[1]]

            vocabulary[size] = merge_bytes
            new_id = size
            size += 1

            stime2 = time.perf_counter()
            # update word_encodings
            for word, word_tokens in word_encodings.items():
                i = 0
                new_tokens = []                
                has_new_id = False
 
                while i < len(word_tokens):
                    if i < len(word_tokens) - 1 and (word_tokens[i], word_tokens[i + 1]) == merge_pair:
                        new_tokens.append(new_id)
                        i += 2
                        has_new_id = True
                    else:
                        new_tokens.append(word_tokens[i])
                        i += 1

                if has_new_id:
                    word_encodings[word] = new_tokens

            etime2 = time.perf_counter()
            update_time += (etime2 - stime2)

            merges.append((vocabulary[merge_pair[0]], vocabulary[merge_pair[1]]))
        print(f"count_pairs_time: {count_pairs_time}, max_time: {max_time}, update_time: {update_time}")
        print(f"total time: {time.perf_counter() - stime}")
        return vocabulary, merges

    @staticmethod    
    def _count_pairs(word_counts, word_encodings, pair_strings, vocabulary):
        pair_counts = defaultdict(int)
        for word, count in word_counts.items():
            encoding = word_encodings[word]
            for i in range(0, len(encoding) - 1):
                pair = encoding[i], encoding[i + 1]
                pair_counts[pair] += count
                if pair not in pair_strings:
                    pair_strings[pair] = (vocabulary[pair[0]], vocabulary[pair[1]])

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

    def _pretokenize_and_count(self, input_path: str, special_tokens: list[str]):
        # pre-compile regex
        pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        # build split pattern
        special_pattern = "|".join(re.escape(token) for token in special_tokens)
        word_counts = defaultdict(int)

        for chunk in BPE_Trainer._chunk_documents_streaming(input_path):
            blocks = re.split(special_pattern, chunk)
            for block in blocks:
                for match in re.finditer(pattern, block):
                    text = match.group(0)
                    word_counts[text] += 1

        return word_counts
    
