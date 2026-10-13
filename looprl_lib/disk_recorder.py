"""
Utilities to manage persistent lists of strings on disk.
"""


import os

import ray


def indexed_elements_in_dir(dir: str) -> list[int]:
    idxs: list[int] = []
    for name in os.listdir(dir):
        try:
            idxs.append(int(name))
        except ValueError:
            pass
    return idxs


@ray.remote
class DiskRecorder:

    def __init__(self, dir: str):
        self.dir = dir
        if not os.path.isdir(self.dir):
            os.makedirs(self.dir, exist_ok=True)
        existing = indexed_elements_in_dir(self.dir)
        self.next_id = 0 if not existing else max(existing) + 1

    def push(self, elements: list[str]) -> list[int]:
        ids=[]
        for e in elements:
            id = self.next_id
            ids.append(id)
            filename = os.path.join(self.dir, str(id))
            with open(filename, 'w') as f:
                f.write(e)
            self.next_id += 1
        return ids
