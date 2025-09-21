#include "max_heap.h"
#include <stdexcept>

enum class ComparisonResult {
    Less,
    Equal,
    Greater
};

ComparisonResult _three_way_compare(const std::vector<int>& a, const std::vector<int>& b) {
    // 1. 获取较小的 vector 的大小
    size_t min_size = std::min(a.size(), b.size());

    // 2. 逐个元素进行比较
    for (size_t i = 0; i < min_size; ++i) {
        if (a[i] < b[i]) {
            return ComparisonResult::Less;
        }
        if (a[i] > b[i]) {
            return ComparisonResult::Greater;
        }
    }

    // 3. 如果所有共同元素都相同，则比较大小
    if (a.size() < b.size()) {
        return ComparisonResult::Less;
    }
    if (a.size() > b.size()) {
        return ComparisonResult::Greater;
    }

    // 4. 如果大小和所有元素都相同
    return ComparisonResult::Equal;
}

bool HeapItem::operator<(const HeapItem& other) const {
    if(count < other.count){
        return true;
    }else if(count == other.count){
        ComparisonResult r1 = _three_way_compare(pair_string[0], other.pair_string[0]);
        if(r1 == ComparisonResult::Greater){
            return false;
        }else if(r1 == ComparisonResult::Less){
            return true;
        }
        ComparisonResult r2 = _three_way_compare(pair_string[1], other.pair_string[1]);
        if(r2 == ComparisonResult::Greater){
            return false;
        }else{ // it's not possible equal again
            return true;
        }
    }else{
        return false;
    }
}


void _siftdown_max(std::vector<HeapItem>& heap, int startpos, int pos) {
    HeapItem newitem = std::move(heap[pos]);
    
    while (pos > startpos) {
        int parentpos = (pos - 1) >> 1;
        HeapItem& parent = heap[parentpos];
        
        if (parent < newitem) {
            heap[pos] = std::move(parent);
            pos = parentpos;
        } else {
            break;
        }
    }  
    heap[pos] = std::move(newitem);
}


void _siftup_max(std::vector<HeapItem>& heap, int pos) {
    int endpos = heap.size();
    int startpos = pos;
    HeapItem newitem = std::move(heap[pos]);
    
    int childpos = 2 * pos + 1;
    
    while (childpos < endpos) {
        int rightpos = childpos + 1;
        if (rightpos < endpos && !(heap[rightpos] < heap[childpos])) {
            childpos = rightpos;
        }
        
        heap[pos] = std::move(heap[childpos]);
        pos = childpos;
        childpos = 2 * pos + 1;
    }
    
    heap[pos] = std::move(newitem);
    
    _siftdown_max(heap, startpos, pos);
}

void heappush(std::vector<HeapItem>& heap, const HeapItem& item) {
    heap.push_back(item);
    
    _siftdown_max(heap, 0, heap.size() - 1);
}

HeapItem heappop(std::vector<HeapItem>& heap) {
    if (heap.empty()) {
        throw std::out_of_range("heappop from empty heap");
    }

    HeapItem lastelt = std::move(heap.back());
    heap.pop_back();

    if (!heap.empty()) {
        HeapItem returnitem = std::move(heap.front());
        heap.front() = std::move(lastelt);
        _siftup_max(heap, 0);
        return returnitem;
    }

    return lastelt;
}

void heapify(std::vector<HeapItem>& x) {
    int n = x.size();
    // 从最后一个非叶子节点开始，向前遍历
    for (int i = n / 2 - 1; i >= 0; --i) {
        _siftup_max(x, i);
    }
}