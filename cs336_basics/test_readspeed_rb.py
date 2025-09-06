import sys
import time
def main():
    if len(sys.argv) < 2:
        print(f"need 2 args: test_readspeed filename")
        return -1

    buffer_size=65536
    if len(sys.argv) > 2:
        buffer_size = int(sys.argv[2])

    start_time = time.perf_counter()


    total_bytes = 0
    with open(sys.argv[1], "rb", buffering=buffer_size) as f:
        while True:
            data = f.read(8192)
            if not data:
                break
            if total_bytes == 0:
                print(f"{type(data)=}")
            total_bytes += len(data)

    end_time = time.perf_counter()
    print(f"time: {end_time - start_time:.2f}s")
    print(f"{total_bytes=}")

if __name__ == '__main__':
    main()