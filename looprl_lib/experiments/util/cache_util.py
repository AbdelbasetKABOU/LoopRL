import os
import pickle
from typing import Callable, TypeVar

T = TypeVar("T")


def cached(file: str, make_data: Callable[[], T]) -> T:
    if os.path.isfile(file):
        with open(file, 'rb') as f:
            return pickle.load(f)
    else:
        data = make_data()
        os.makedirs(os.path.dirname(file), exist_ok=True)
        with open(file, 'wb') as f:
            pickle.dump(data, f)
        return data
