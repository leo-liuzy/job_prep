class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        # Solution O(n): prefix-suffix trick
        prefix_prod = []
        prod = 1
        for i, n in enumerate(nums):
            if i > 0:
                prod *= nums[i - 1]
            prefix_prod.append(prod)

        prod = 1
        suffix_prod = []
        for i in range(len(nums) - 1, -1, -1):
            if i < len(nums) - 1:
                prod *= nums[i + 1]
            suffix_prod.insert(0, prod)

        ret = []
        for i in range(len(nums)):
            ret.append(suffix_prod[i] * prefix_prod[i])
        return ret

        # prefix-suffix trick without axuiliary space
        # ret = []
        # prod = 1
        # for i, n in enumerate(nums):
        #     if i > 0:
        #         prod *= nums[i-1]
        #     ret.append(prod)

        # prod = 1
        # for i in range(len(nums)-1, -1, -1):
        #     if i < len(nums) - 1:
        #         prod *= nums[i+1]
        #     ret[i] *= prod
        # return ret
