from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Union

StylingFun = Callable[[str], str]

StyledStr = Union[str, tuple[str, StylingFun]]


@dataclass
class Column:
    header: str
    data: list[StyledStr]
    min_width: Optional[int] = None
    max_width: Optional[int] = None
    align_right: bool = False


def force_range(x: int, lb: Optional[int], ub: Optional[int]) -> int:
    if lb is not None and x < lb:
        x = lb
    if ub is not None and x > ub:
        x = ub
    return x


def fit_str(s: str, width: int, align_right: bool):
    if len(s) < width:
        pad = (width - len(s)) * " "
        return pad + s if align_right else s + pad
    return s


def no_style(x: str): return x

def compose_styles(style_1, style_2): return lambda s: style_1(style_2(s))

def split_styled(styled: StyledStr) -> tuple[str, StylingFun]:
    if isinstance(styled, str):
        return styled, no_style
    else:
        return styled

def styled_len(styled: StyledStr) -> int:
    return len(split_styled(styled)[0])


def pp_table(
        cols: list[Column],
        show_headers: bool = True,
        header_styling: StylingFun = no_style,
        header_bot_margin: int = 0,
        cols_sep: int = 2,
        displayed_rows: Optional[Iterable[int]] = None,
    ) -> str:
    if not cols: return ""
    num_lines = len(cols[0].data)
    assert all(len(c.data) == num_lines for c in cols)
    data = [[c.data[i] for c in cols] for i in range(num_lines)]
    # We start by adding the headers to the data
    if show_headers:
        data.insert(0, [(c.header, header_styling) for c in cols])
        if displayed_rows is not None:
            displayed_rows = [0] + [r + 1 for r in displayed_rows]
    # Then we compute column widths
    ncols = len(cols)
    max_content_len = [
        max(styled_len(row[c]) for row in data) for c in range(ncols)]
    col_width = [
        force_range(m, spec.min_width, spec.max_width)
        for spec, m in zip(cols, max_content_len)]
    s = ""
    rows = range(len(data)) if displayed_rows is None else displayed_rows
    for r in rows:
        for c in range(ncols):
            content, styling = split_styled(data[r][c])
            content = fit_str(content, col_width[c], cols[c].align_right)
            s += styling(content)
            if c < ncols - 1:
                s += cols_sep * " "
        s += "\n"
        if show_headers and r == 0:
            s += "\n" * header_bot_margin
    return s