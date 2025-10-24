def knn_predict(X_train, y_train, x, k=3, method="argsort"):
    """
    最简单面试版 KNN 实现 (支持 argsort / heapq / argpartition 三种取 Top-K 方法)
    Args:
    X
    _
    train: 训练样本矩阵 [N, D]
    y_
    train: 标签 [N]
    x: 单个测试样本 [D]
    k: 近邻数量
    method: 选择Top-K方法 ("argsort"
    ,
    "heapq"
    ,
    "argpartition")
    """

    # 󾠮 计 算 欧式距离 (可 以 改 成余 弦距离 等 )
    dist = np.sqrt(np.sum((X_train - x) ** 2, axis=1))

    # 󾠯 取 前 k 个 最小 距离 的 索 引
    if method == "argsort":
        # 完 全排 序 → O(n log n)
        # ✅ 简 单 直 接（适 合小数据集 ）
        idx = np.argsort(dist)[:k]
    elif method == "heapq":
        # 维护大小 为 k 的 最大堆 → O(n log k)
        # ✅ 适 合大数据、 k 很小 的 情况
        idx = heapq.nsmallest(k, range(len(dist)), key=lambda i: dist[i])
    else:
        raise ValueError("method must be one of ['argsort'
'argpartition']")

    labels = y_train[idx]
    pred = np.bincount(labels).argmax()

    return pred