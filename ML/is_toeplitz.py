def is_toeplitz(matrix: List[List[int]]) -> bool:
    """
    判断给定矩阵是否为 Toeplitz 矩阵。
    返回 True 表示是，False 表示否。
    """
    # TODO: 实 现 算 法
    # 要 点 ：
    # 1) 空 矩 阵 、 单行或 单 列 -> True
    # 2) 如 有任意一行的长
    度 与 第一行不 同 -> False（非 规则 矩 阵 ）
    # 3) 遍历 i:1..m-1, j:1..n-1 ， 检查 matrix[i][j] == matrix[i-1][j-1]
    # 4) 全 部满足 即 为 True
    m = len(matrix)
    if m == 1 or m == 0:
        return True
    
    n = len(matrix[0])
    
    for i in range(1, m):
        if len(matrix[i]) != len(matrix[i-1]):
            return False
    
    for i in range(1, m):
        for j in range(1, n):
            if matrix[i][j] == matrix[i-1][j-1]:
                continue
            else:
                return False
    return True