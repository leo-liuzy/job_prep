from typing import List


# 差分数组工具类
class Difference:
    # 差分数组
    def __init__(self, nums: List[int]):
        assert len(nums) > 0
        self.diff = [0] * len(nums)
        # 根据初始数组构造差分数组
        self.diff[0] = nums[0]
        for i in range(1, len(nums)):
            self.diff[i] = nums[i] - nums[i - 1]

    # 给闭区间 [i, j] 增加 val（可以是负数）
    def increment(self, i: int, j: int, val: int) -> None:
        self.diff[i] += val
        if j + 1 < len(self.diff):
            self.diff[j + 1] -= val

    # 返回结果数组
    def result(self) -> List[int]:
        res = [0] * len(self.diff)
        # 根据差分数组构造结果数组
        res[0] = self.diff[0]
        for i in range(1, len(self.diff)):
            res[i] = res[i - 1] + self.diff[i]
        return res


if __name__ == "__main__":
    a = Difference([8, 2, 6, 3, 1])
