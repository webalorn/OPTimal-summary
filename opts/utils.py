def min_with_none(values):
    values = [v for v in values if v is not None]
    if values:
        return min(values)
    return None
    