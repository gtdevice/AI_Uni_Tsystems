def bin_search(data, x):
    l = 0
    r = len(data) - 1
    while l <= r:
        mid = l + (r - l) // 2
        if data[mid] == x:
            return mid
        elif data[mid] < x:
            l = mid + 1
        else:
            r = mid - 1
    return -1 #in case the result isn't in data set
