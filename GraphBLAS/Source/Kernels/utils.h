__device__ int binarySearch(const int* array,
                              int        target,
                              int        begin,
                              int        end) {
  while (begin < end) {
    int mid  = begin + (end - begin) / 2;
    int item = array[mid];
    if (item == target)
      return mid;
    bool larger = (item > target);
    if (larger)
      end = mid;
    else
      begin = mid + 1;
  }
  return -1;
}
