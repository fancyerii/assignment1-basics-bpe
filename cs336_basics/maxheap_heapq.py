import heapq
# https://stackoverflow.com/questions/33024215/built-in-max-heap-api-in-python

def heappush(heap, item):
    heap.append(item)
    heapq._siftdown_max(heap, 0, len(heap)-1)


def heappop(heap):
    """Maxheap version of a heappop."""
    lastelt = heap.pop()  # raises appropriate IndexError if heap is empty
    if heap:
        returnitem = heap[0]
        heap[0] = lastelt
        heapq._siftup_max(heap, 0)
        return returnitem
    return lastelt

def heapify(heap):
    heapq._heapify_max(heap)