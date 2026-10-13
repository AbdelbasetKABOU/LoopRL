#!/bin/sh
rm -rf _build .mypy_cache .pytest_cache
find . | grep -E "__pycache__" | xargs rm -rf