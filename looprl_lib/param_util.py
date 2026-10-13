"""
Utilities to manage nested parameter dataclasses.
"""

import dataclasses
from dataclasses import dataclass
from typing import Any, Optional

from dataclasses_json import dataclass_json


def paths_candidates(dirs, field):
    yield ".".join([n for n, _ in dirs] + [field])
    for i, (_, t) in enumerate(dirs):
        if t.short_name is not None:
            yield ".".join(
                ["::"+t.short_name] + [n for n, _ in dirs[i+1:]] + [field])


def from_diff(name: Optional[str] = None):
    def decorator(dataclass):
        def build_rec(config, default, dirs, used):
            args = {}
            for f in dataclasses.fields(dataclass):
                found = False
                for path in paths_candidates(dirs, f.name):
                    if path in config:
                        value = config[path]
                        used.add(path)
                        found = True
                        break
                if not found:
                    value = getattr(default, f.name)
                    if hasattr(f.type, 'from_diff_rec'):
                        value = f.type.from_diff_rec(
                            config, value, dirs + [(f.name, f.type)], used)
                args[f.name] = value
            return dataclass(**args)
        def update(self, diff):
            used = set()
            default = self
            res = dataclass.from_diff_rec(diff, default, [], used)
            for k in diff.keys():
                assert k in used, f"Unused config item: {k}"
            return res
        def build(diff):
            return dataclass().update(diff)
        dataclass.from_diff_rec = staticmethod(build_rec)
        dataclass.from_diff = staticmethod(build)
        dataclass.update = update
        dataclass.short_name = name
        return dataclass
    return decorator


# Not used in practice since it does not work well with mypy
def params(name: Optional[str] = None):
    def decorator(cls):
        cls = dataclasses.dataclass(frozen=True)(cls)
        cls = dataclass_json(cls)
        cls = from_diff(name)(cls)
        return cls
    return decorator


if __name__ == '__main__':

    @from_diff(name='point')
    @dataclass(frozen=True)
    class Point:
        x: int = 0
        y: int = 0

    @from_diff(name='rect')
    @dataclass(frozen=True)
    class Rectangle:
        upper_left: Point = Point()
        lower_right: Point = Point(1, 1)

    @from_diff()
    @dataclass(frozen=True)
    class Config:
        rect: Rectangle = Rectangle()
        @staticmethod
        def from_diff(d: dict[str, Any]) -> 'Config': ...

    configs: list[dict] = [
        {},
        {'::rect.upper_left.y': 42},
        {'::point.x': 5, 'rect.upper_left.y': 3}
    ]
    for c in configs:
        print(Config.from_diff(c))
