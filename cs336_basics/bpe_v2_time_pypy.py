import regex as re
from collections import defaultdict
import time

CHUNK_SIZE = 1024 *  50

class BPE_Trainer():
    def train(self, input_path, vocab_size, special_tokens, *args):
        start_time = time.perf_counter()
        word_counts = self._pretokenize_and_count(input_path, special_tokens)
        end_time = time.perf_counter()
        print(f"_pretokenize_and_count time: {end_time - start_time}")
        print(f"word_counts: {len(word_counts)}")
        vocabulary = {}
        merges = []
        return vocabulary, merges
    

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
        split_time = 0
        match_time = 0
        text_len = 0
        for chunk in BPE_Trainer._chunk_documents_streaming(input_path):
            start_time = time.perf_counter()
            blocks = re.split(special_pattern, chunk)
            end_time = time.perf_counter()
            split_time += (end_time - start_time)
            for block in blocks:
                start_time = time.perf_counter()
                for match in re.finditer(pattern, block):
                    text = match.group(0)
                    text_len += len(text)
                end_time = time.perf_counter()
                match_time += (end_time - start_time)

        print(f"{split_time=}, {match_time=}, {text_len=}")
        return word_counts



if __name__ == '__main__':
    bpe_trainer = BPE_Trainer()
    
    start_time = time.perf_counter()
    data_path = 'data/TinyStoriesV2-GPT4-train.txt'
    vocab_size = 10000
    vocabulary, merges = bpe_trainer.train(data_path, vocab_size, 
                                           ["<|endoftext|>"])
    end_time = time.perf_counter()
    print(f"total train time: {end_time - start_time:.2f} seconds")    