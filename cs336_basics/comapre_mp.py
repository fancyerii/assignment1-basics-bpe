import cs336_basics.bpe_v1 as v1
import cs336_basics.bpe_v3 as v3


if __name__ == '__main__':
    t1 = v1.BPE_Trainer()
    t3 = v3.BPE_Trainer()
    data_path = "./data/owt_train.txt"
    wc1 = t1._pretokenize_and_count(data_path, special_tokens = ["<|endoftext|>"])
    wc3 = t3._pretokenize_and_count_mp(data_path, special_tokens = ["<|endoftext|>"],
                                            num_counter=32,
                                            num_merger=4,
                                            do_monitor=True)
    if wc1 != wc3:
        print("bug")


