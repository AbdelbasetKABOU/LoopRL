"""
Convert the Code2Inv benchmark examples to Looprl syntax.
"""


import os
import parse_bench
import looprl

BENCH_DIR = os.path.join('benchmarks', 'code2inv', 'bench')
ORIG_DIR = os.path.join(BENCH_DIR, 'original')
CONVERTED_DIR = os.path.join(BENCH_DIR, 'converted')


# Transform "if P {assert Q;}" into "assume P; assert Q;"


def convert_prog(s: str) -> str:
    s = parse_bench.parse(s)
    return str(looprl.Prog(s).normalize_task())


if __name__ == "__main__":
    assert os.path.isdir(ORIG_DIR), f"Missing dir: {ORIG_DIR}"
    if not os.path.isdir(CONVERTED_DIR):
        os.mkdir(CONVERTED_DIR)
    for f in os.listdir(ORIG_DIR):
        name, ext = os.path.splitext(f)
        file = os.path.join(ORIG_DIR, f)
        if ext == ".c":
            content = open(file, 'r').read()
            try:
                converted = convert_prog(content)
                dest_file = os.path.join(CONVERTED_DIR, name + ".txt")
                open(dest_file, 'w').write(converted)
            except Exception as e:
                print(f"Error while converting: {f}")
                raise