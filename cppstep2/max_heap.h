#pragma once
#include <vector>

struct HeapItem{
    int count;
    std::vector<std::vector<int>> pair_string;
    std::pair<int,int> pair;

    bool operator<(const HeapItem& other) const;
};

void heappush(std::vector<HeapItem>& heap, const HeapItem& item);

HeapItem heappop(std::vector<HeapItem>& heap);

void heapify(std::vector<HeapItem>& heap);