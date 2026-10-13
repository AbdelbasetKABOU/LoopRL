# Utility functions to prettify probes

import re

from ansi.color.fg import yellow  # type: ignore
from ansi.color.fx import bold, underline  # type: ignore


def whole_id(regex):
    id_char = r"[a-zA-Z0-9_\-\.]"
    return f"(?<!{id_char})(" + regex + f")(?!{id_char})"

def highlight_keywords(s: str, style=bold):
    kwds = [
        'while', 'if', 'else', 'assume', 'assert', 'invariant',
        'goal', 'probe', 'constraint', 'properties', 'inv-candidate',
        'sample-spec']
    regex = "|".join([re.escape(k) for k in kwds])
    regex = whole_id(regex)
    def replace(m):
        return style(m[0])
    return re.sub(regex, replace, s)

def with_unicode_operators(s: str) -> str:
    unicode = {
        '->': '⟶ ', '>=': '≥', '<=': '≤', '||': '∨', '&&': '∧',
        '!=': '≠', '==': '='}
    for (before, after) in unicode.items():
        s = s.replace(before, after)
    return s

def highlight_code_annots(s: str, style=yellow):
    regex = r"'[^']*'"
    return re.sub(regex, lambda m: style(m[0]), s)

def highlight_to_prove(s: str, style=underline):
    return re.sub('to-prove', lambda m: style(m[0]), s)

def prettify(s: str):
    s = with_unicode_operators(s)
    s = highlight_keywords(s)
    # s = highlight_code_annots(s)
    s = highlight_to_prove(s)
    return s

def highlight_focus(s: str, focused: str, style=yellow):
    regex = whole_id(re.escape(focused))
    return re.sub(regex, lambda m: style(m[0]), s)
