def jaccard_similarity(a: Iterable[Hashable], b: Iterable[Hashable]) -> float:
    """
    计算输入 a 与 b 的 Jaccard 相似度。
    TODO：在此处实现函数逻辑。
    要点：
    - sa, sb = set(a), set(b)
    - 若二者均为空，则返回 1.0
    - 返回 len(sa & sb) / len(sa | sb) 的浮点结果
    """
    sa = set(a)
    sb = set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)