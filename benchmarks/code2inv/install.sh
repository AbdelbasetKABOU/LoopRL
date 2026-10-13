#!/bin/sh

git clone git@github.com:PL-ML/code2inv.git
mkdir ./benchmarks/code2inv/bench/
mkdir ./benchmarks/code2inv/bench/original
cp code2inv/benchmarks/C_instances/c/* ./benchmarks/code2inv/bench/original
rm ./benchmarks/code2inv/bench/original/test.c
rm -rf code2inv
python benchmarks/code2inv/convert_bench.py