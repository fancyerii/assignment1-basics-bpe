from cs336_basics.bpe_v3_step1_result import BPE_Trainer
import time
import sys
def main():
    out_dir = "./data/openwebdump"
    if len(sys.argv) > 1:
        out_dir = sys.argv[1]
    print(f"{out_dir=}")
    data_path = "./data/owt_train.txt"
    if len(sys.argv) > 2:
        data_path = sys.argv[2]
    print(f"{data_path=}")
    
    bpe_trainer = BPE_Trainer()
    vocab_size = 32000
    start_time = time.perf_counter()
    bpe_trainer.dump_step1_results(data_path, vocab_size, 
                                           special_tokens = ["<|endoftext|>"],
                                           output_dir=out_dir,
                                           num_counter=8,
                                           num_merger=1,
                                           do_monitor=True,
                                           num_max_processes=32)
    end_time = time.perf_counter()
    print(f"openwebtext time: {end_time - start_time:.2f} seconds")
 

if __name__ == '__main__':
    main()