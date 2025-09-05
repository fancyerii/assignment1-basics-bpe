from cs336_basics.bpe_v3_step1_result import BPE_Trainer
import time
def main():
    data_path = "./data/TinyStoriesV2-GPT4-train.txt"
    bpe_trainer = BPE_Trainer()
    vocab_size = 10000
    start_time = time.perf_counter()
    bpe_trainer.dump_step1_results(data_path, vocab_size, 
                                           special_tokens = ["<|endoftext|>"],
                                           output_dir="./data/tinystorydump",
                                           num_counter=8,
                                           num_merger=1,
                                           do_monitor=True,
                                           num_max_processes=8)
    end_time = time.perf_counter()
    print(f"TinyStories time: {end_time - start_time:.2f} seconds")

if __name__ == '__main__':
    main()