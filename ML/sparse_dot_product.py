from typing import List, NamedTuple

class IndexValue(NamedTuple):
    index: int
    value: float

def sparse_dot_product(a: List[IndexValue], b: List[IndexValue]) -> float:
    i = j = 0
    result = 0.0
    while i < len(a) and j < len(b):
        if a[i].index == b[j].index:
            result += a[i].value * b[j].value
            i += 1
            j += 1
        elif a[i].index < b[j].index:
            # a[i] has smaller index, move pointer i forward
            i += 1
        else:
            # b[j] has smaller index, move pointer j forward
            j += 1
    return result
