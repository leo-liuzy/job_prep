from typing import List


def combinationSum2(candidates: List[int], target: int) -> List[List[int]]:
    def dfs(i, s):
        if s == 0:
            ans.append(t[:])
            return
        if i >= len(candidates) or s < candidates[i]:
            return

        for j in range(i, len(candidates)):
            if j > i and candidates[j] == candidates[j - 1]:
                continue
            t.append(candidates[j])
            dfs(j + 1, s - candidates[j])
            t.pop()

    candidates = sorted(candidates)
    t = []
    ans = []
    dfs(0, target)

    return ans


print(
    combinationSum2(
        candidates=[10, 1, 2, 7, 6, 1, 5],
        target=8,
    )
)
