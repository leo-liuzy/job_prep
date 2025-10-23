def gcdOfStrings(str1: str, str2: str) -> str:
    shorter_s = str1 if len(str1) < len(str2) else str2
    longer_s = str2 if len(str1) < len(str2) else str1

    def check_division(s, t):
        return s.replace(s, t) == ""

    p = len(shorter_s)
    while p > 0:
        x = shorter_s[:p]
        if check_division(shorter_s, x) and check_division(longer_s, x):
            return x
        p -= 1
    return x


print(gcdOfStrings("ABCABC", "ABC"))
