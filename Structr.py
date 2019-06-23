class Structr:
    def __init__(self):
        self.el = []

    def add_elem(self, element):
        if not (self.is_number(element)):
            return False
        index = 0
        if len(self.el) > 0:
            index = self.bin_search_place(element)
        self.el.insert(index, float(element))
        return True

    def del_max(self):
        return self.el.pop()
#List pop intermediate operation O(k), in case 0 element, it's O(1)
    def del_min(self):
        return self.el.pop(0)

    def rem_elem(self, element):
        self.el.remove(element)

    def is_number(self, s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    def bin_search_place(self, x):
        l = 0
        r = len(self.el) - 1
        mid = 0
        while l <= r:
            mid = l + (r - l) // 2
            if self.el[mid] == x:
                return mid
            elif self.el[mid] < x:
                l = mid + 1
            else:
                r = mid - 1
        if x < self.el[mid]:
            return mid
        else:
            return mid + 1
