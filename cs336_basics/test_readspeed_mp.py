import multiprocessing as mp
import os
import sys
import time

# 假设你的文件是按行处理的，这个函数找到每个分块的起始和结束行
def get_file_offsets(file_path, num_processes):
    file_size = os.path.getsize(file_path)
    print(f"{file_size=}")
    chunk_size = file_size // num_processes
    offsets = [0]
    with open(file_path, 'r') as f:
        for i in range(1, num_processes):
            f.seek(chunk_size * i)
            # 找到下一个换行符，确保按行分割
            f.readline()
            offsets.append(f.tell())
    offsets.append(file_size)
    return offsets

def process_chunk(file_path, start_offset, end_offset):
    # 每个进程处理一个文件分块
    print(f"Process {os.getpid()} reading from {start_offset} to {end_offset}")
    total_chars = 0
    total_line = 0
    with open(file_path, 'r') as f:
        f.seek(start_offset)

        while f.tell() < end_offset:
            line = f.readline()
            if not line:
                break
            total_chars += len(line)
            total_line += 1

        if f.tell() != end_offset:
            print(f"bug f.tell={f.tell()}, {end_offset=}")

    print(f"Process {os.getpid()} read {total_chars=}, {total_line=}")

def main(file_path, num_processes):
    start_time = time.perf_counter()
    offsets = get_file_offsets(file_path, num_processes)
    processes = []
    
    for i in range(num_processes):
        start = offsets[i]
        end = offsets[i+1]
        p = mp.Process(target=process_chunk, args=(file_path, start, end))
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()
    end_time = time.perf_counter()
    print(f"total time: {end_time - start_time:.2f}s")
        
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"need 3 args: file_path num_processes")
        sys.exit(-1)
    file_path = sys.argv[1]
    num_processes = int(sys.argv[2])  # 你可以根据CPU核心数调整
            
    main(file_path, num_processes)