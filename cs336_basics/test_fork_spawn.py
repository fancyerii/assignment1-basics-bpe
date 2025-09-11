import time, multiprocessing as mp

def worker(lst):
    t0 = time.perf_counter()
    for x in lst:          
        pass
    print(f"traverse time: {time.perf_counter() - t0:.2f}s")

if __name__ == '__main__':
    mp.set_start_method('spawn')         
    lst = list(zip(range(5_000_000), range(5_000_000)))
    start_time = time.perf_counter()
    p = mp.Process(target=worker, args=(lst,))
    p.start()
    p.join()
    end_time = time.perf_counter()
    print(f"total time: {end_time - start_time:.2f}s")
