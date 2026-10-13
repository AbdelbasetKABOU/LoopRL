"""
Parse the Code2Inv C benchmark instances using lark.
"""


import lark
from lark import Lark


grammar = r"""
    ?program: "int" "main" "(" ")" instr

    ?instr: "{" instr* "}" -> block
          | "int" name ("," name)* ";" -> declaration
          | "int" name "=" lin ";" -> assignment
          | "(" name "=" lin ")" ";" -> assignment
          | name "=" lin ";" -> assignment
          | name "+=" lin ";" -> incr_assignment
          | "assume" fml ";" -> assumption
          | "assert" fml ";" -> assertion
          | "while" fml instr -> loop
          | "if" fml instr -> if_statement
          | "if" fml instr "else" instr -> if_else_statement

    ?fml: "(" fml ")"
        | "unknown" "(" ")" -> unknown
        | lin comp_op lin -> comparison

    ?lin: linelt
         | lin "+" linelt -> add
         | lin "-" linelt -> sub

    ?linelt: atom
           | "-" atom -> neg

    ?atom: number          -> const
         | number "*" atom -> mul
         | name            -> var
         | "(" lin ")"

    ?comp_op: "="   -> eq
            | "=="  -> eq
            | "<>"  -> ne
            | "!="  -> ne
            | "<="  -> le
            | ">="  -> ge
            | "<"   -> lt
            | ">"   -> gt

    name: CNAME
    number: NUMBER

    COMMENT: "//" /[^\n]/*

    %import common.CNAME
    %import common.NUMBER
    %import common.WS

    %ignore COMMENT
    %ignore WS
"""


@lark.v_args(inline=True)
class ProcessTree(lark.Transformer):
    # Comparisons
    comparison = lambda self, lhs, op, rhs: f"{lhs} {op} {rhs}"
    unknown = lambda self: "??"
    eq = lambda self: "=="
    ne = lambda self: "!="
    le = lambda self: "<="
    lt = lambda self: "<"
    ge = lambda self: ">="
    gt = lambda self: ">"
    # Terms
    add = lambda self, c, l: f"{{{c}}} + {{{l}}}"
    sub = lambda self, c, l: f"{{{c}}} - {{{l}}}"
    mul = lambda self, c, l: f"{c} * {{{l}}}"
    neg = lambda self, t: f"-{{{t}}}"
    var = lambda self, s: s
    const = lambda self, i: str(i)
    name = str
    number = int
    # Programs
    block = lambda self, *progs: " ".join(progs)
    assertion = lambda _, fml: f"assert {fml};"
    assumption = lambda _, fml: f"assume {fml};"
    assignment = lambda _, var, expr: f"{var} = {expr};"
    incr_assignment = lambda _, var, expr: f"{var} = {var} + {expr};"
    loop = lambda _, cond, body: f"while ({cond}) {{{body}}}"
    if_statement = lambda _, c, tb: f"if ({c}) {{{tb}}}"
    if_else_statement = lambda _, c, tb, fb: f"if ({c}) {{{tb}}} else {{{fb}}}"
    declaration = lambda _, *args: ""  # type: ignore


parser = Lark(
    grammar,
    start='program',
    parser='lalr',
    transformer=ProcessTree())


def parse(s: str) -> str:
    return parser.parse(s)  # type: ignore