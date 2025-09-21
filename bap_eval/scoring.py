def compute_pei(bias, accuracy, politeness, weights=(1/3, 1/3, 1/3)) -> float:
    """
    Compute Prompt Equity Index (PEI) as a weighted average.

    Args:
        bias (float): Bias score [0–1].
        accuracy (float): Accuracy score [0–1].
        politeness (float): Politeness score [0–1].
        weights (tuple): Weights for (bias, accuracy, politeness).

    Returns:
        float: Composite PEI score.
    """
    w1, w2, w3 = weights
    return w1 * bias + w2 * accuracy + w3 * politeness
