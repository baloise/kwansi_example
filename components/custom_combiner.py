def custom_combine(scores):
    # Weighted combination of first two scores, multiplied by the other two
    weights = [0.7, 0.3]
    weighted_sum = sum(scores[i][1] * weights[i] for i in range(2))
    return weighted_sum * scores[2][1] * scores[3][1]
