from os import path
from typing import Iterable

import looprl

CODE2INV_BENCH_DIR = path.abspath(path.join(
    path.dirname(path.realpath(__file__)),
    '..', 'benchmarks', 'code2inv', 'bench', 'converted'))

# Problems in which a bug has been found and which are therefore excluded
# Note: 72 and 75 are difficult but valid
CODE2INV_INVALID = [
    26, 27,   # what if n = 0?
    61, 62,   # take n = 1 and do one loop iter with branch (c!=n)
    106,      # a<=m instead of a>=m
    72, 75,   # TODO: valid but proof too large for our system
    # 35, 36, 50, 51, 132  # TODO: solved in 0 steps
]

# Probe size of 80 not enough for 72 and 75
# The next one is 93 which needs ... tokens

def code2inv(i: int) -> looprl.Prog:
    with open(path.join(CODE2INV_BENCH_DIR, f"{i}.txt")) as f:
        return looprl.Prog(f.read())

def iter_code2inv() -> Iterable[tuple[int, looprl.Prog]]:
    for i in range(1, 134):
        if i not in CODE2INV_INVALID:
            yield (i, code2inv(i))

EXAMPLES: dict[str, str] = {}

EXAMPLES['strengthen'] = """
assume x > 0;
while (x < 0) {
    if (??) {
        x = x + 1;
    } else {
        x = x - n;
    }
}
assume x >= 0;
assert x > 0;
"""

EXAMPLES['paper-example'] = """
assume x >= 1;
y = 0;
while (y < 1000) {
    x = x + y;
    y = y + 1;
}
assert x >= y;
"""
