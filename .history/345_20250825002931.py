def reverseVowels(s: str) -> str:
    letters = list(s)
    left = 0
    right = len(s) - 1
    vowels = ["a", "e", "i", "o", "u"]

    def is_vowel(l):
        return l.lower() in vowels

    while left < right:
        if is_vowel(letters[left]) and is_vowel(letters[right]):
            tmp = letters[left]
            letters[left] = letters[right]
            letters[right] = tmp
        elif is_vowel(letters[left]):
            right -= 1
        else:
            left += 1
    return "".join(letters)


print(reverseVowels("hello"))
