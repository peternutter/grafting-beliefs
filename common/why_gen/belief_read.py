import math

K = 1.0


def p_real(row):
    return 1.0 / (1.0 + math.exp(-row["logdiff"] * K))


def p_real_total(row):
    d = row.get("logdiff_sum")
    return None if d is None else 1.0 / (1.0 + math.exp(-d * K))


def anchor_mass(row):
    return row.get("anchor_mass")


def schema(row):
    return row.get("schema", 1)
