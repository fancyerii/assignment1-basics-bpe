import sys
import time
def main():
    if len(sys.argv) < 2:
        print(f"need 2 args: test_readspeed filename")
        return -1
    
    total_chars = 0
    start_time = time.perf_counter()
    with open(sys.argv[1]) as f:
        for line in f:
            total_chars += len(line)
    end_time = time.perf_counter()
    print(f"time: {end_time - start_time:.2f}s")
    print(f"{total_chars=}")

if __name__ == '__main__':
    main()