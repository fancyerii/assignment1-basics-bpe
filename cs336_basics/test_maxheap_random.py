from cs336_basics import maxheap_py
import heapq
import random
from cs336_basics import maxheap_heapq
from cs336_basics import maxheapq 

def main():
    N = 1_000
    M = 100_000
    random.seed(1234)
    for _ in range(N): 
        arr = [random.randint(-100_000_000, 100_000_000) for _ in range(M)]
        arr_copy = [-x for x in arr]
        arr_copy2 = arr_copy[:]
        arr_copy3 = arr_copy[:]

        added = [random.randint(-100_000_000, 100_000_000) for _ in range(100)]

        heapq.heapify(arr)
        maxheap_py.heapify(arr_copy)
        maxheap_heapq.heapify(arr_copy2)
        maxheapq.heapify(arr_copy3)
        assert arr == [-x for x in arr_copy]
        for z in added:
            x1 = heapq.heappop(arr)
            x2 = maxheap_py.heappop(arr_copy)
            x3 = maxheap_heapq.heappop(arr_copy2)
            x4 = maxheapq.heappop(arr_copy3)
            assert x1 == -x2
            assert x2 == x3
            assert x3 == x4
            heapq.heappush(arr, z)
            maxheap_py.heappush(arr_copy, -z)
            maxheap_heapq.heappush(arr_copy2, -z)
            maxheapq.heappush(arr_copy3, -z)
            

if __name__ == '__main__':
    main()
