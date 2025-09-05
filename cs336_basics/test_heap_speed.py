import cs336_basics.maxheap_py as maxheap_py
import cs336_basics.maxheap_heapq as maxheap_heapq
import cs336_basics.maxheapq as maxheapq
import random
import time

def main():
    N = 1_000
    M = 100_000

    time1 = 0
    time2 = 0
    time3 = 0
    for _ in range(N): 
        arr = [random.randint(-100_000_000, 100_000_000) for _ in range(M)]
        arr_copy = arr[:]
        arr_copy2 = arr[:]

        added = [random.randint(-100_000_000, 100_000_000) for _ in range(100)]

        stime = time.time()
        maxheap_py.heapify(arr)
        for z in added:
            x = maxheap_py.heappop(arr)
            maxheap_py.heappush(arr, z)
        etime = time.time()
        time1 +=(etime - stime)

        stime = time.time()
        maxheap_heapq.heapify(arr_copy)
        for z in added:
            x = maxheap_heapq.heappop(arr_copy)
            maxheap_heapq.heappush(arr_copy, z)
        etime = time.time()
        time2 +=(etime - stime)

        stime = time.time()
        maxheapq.heapify(arr_copy2)
        for z in added:
            x = maxheapq.heappop(arr_copy2)
            maxheapq.heappush(arr_copy2, z)
        etime = time.time()
        time3 +=(etime - stime)

    print(f"maxheapq_py: {time1}, maxheap_heapq: {time2}, maxheapq: {time3}")

if __name__ == '__main__':
    main()