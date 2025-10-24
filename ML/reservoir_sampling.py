def reservoir_sampling(data, k):
    reservoir = []
    for i, item in enumerate(stream):
        if i < k:
            reservoir.append(item)
        else:
            # Pr(item i enter reservoir) = k / i
            j = random.randint(0, i)
            # Pr(item i in the reservoir survice all later step)
            # = \prod_{j=i+1}^N (1 - 1/j) = i / N
            # Pr(item i stays) = (k / i) * (i / N) = k / N
            if j < k: 
                reservoir[j] = item
    return reservoir