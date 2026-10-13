import traceback
from collections import deque
from typing import Callable, TypeVar

from ansi.color.fg import boldyellow as emph  # type: ignore
from ansi.color.fg import red, yellow  # type: ignore
from ansi.color.fx import bold  # type: ignore
from looprl import SearchTree

from .prettify import highlight_focus, prettify

META_FOCUS = "focus"

T = TypeVar("T")

ActionsLog = list[tuple[int, str]]


def explore(
    st: SearchTree[T],
    show_weights=False,
    show_success: Callable[[T], str] = lambda x: str(x)
) -> tuple[SearchTree[T], ActionsLog]:
    history = [st]
    actions: ActionsLog = []
    while True:
        print("\n" * 100)
        if st.is_choice():
            print(emph("Probe:"), "\n")
            probe = prettify(repr(st.probe()))
            meta = st.probe().meta()
            if META_FOCUS in meta:
                probe = highlight_focus(probe, meta[META_FOCUS], style=red)
                del meta[META_FOCUS]
            print(probe, "\n")
            for (k, v) in  meta.items():
                print(emph(k[0].upper() + k[1:] + ":"), end="\n\n")
                print(prettify(v), end="\n\n")
            print(emph("Choices:"), "\n")
            for (i, choice) in enumerate(st.choices()):
                weight = f"[{st.weights()[i]:.2f}] " if show_weights else ""
                num = f"[{yellow(str(i))}]"
                print(f"{num} {weight}{prettify(repr(choice))}")
            print("")
        elif st.is_success():
            print(emph("Success"), "\n")
            val = st.success_value()
            print(prettify(show_success(val)), "\n")
        elif st.is_failure():
            print(emph("Failure"), "\n")
            print(st.failure_message(), "\n")
        elif st.is_message():
            print(emph("Message"), "\n")
            print(str(st.message()), "\n")
            _ = input()
            st = st.next()
            continue
        elif st.is_event():
            print(emph(f"Event: {st.event_code()}"), "\n")
            st = st.next()
            continue
        else:
            assert False
        inp = input(emph("> "))
        if not inp:
            return (st, actions)
        elif inp == 'x':  # undo command
            if actions:
                actions.pop()
                st = history.pop()
        elif st.is_choice():
            try:
                i = int(inp)
                assert i >= 0 and i < len(st.choices())
            except Exception:
                pass
            try:
                old_st = st
                st = st.select(i)
                history.append(old_st)
                actions.append((i, str(old_st.choices()[i])))
            except ValueError:
                print("")
                print(red(traceback.format_exc()))
                _ = input()
    assert False


def write_actions_log(log: ActionsLog, file: str) -> None:
    with open(file, 'w') as f:
        for (i, s) in log:
            print(f"{i}: {s}", file=f)


def read_actions_log(file: str) -> ActionsLog:
    log: ActionsLog = []
    with open(file, 'r') as f:
        for s in f.readlines():
            num, cmd = s.split(": ", 1)
            log.append((int(num), cmd.strip()))
    return log


def replay(
    st: SearchTree[T],
    log: ActionsLog
) -> tuple[SearchTree[T], ActionsLog]:
    todo = deque(log.copy())
    while todo:
        i, s = todo.popleft()
        try:
            while st.is_message() or st.is_event():
                st = st.next()
            idxs = [i for (i, c) in enumerate(st.choices()) if str(c) == s]
            if len(idxs) != 1:
                assert False
            st = st.select(idxs[0])
        except:
            todo.appendleft((i, s))
            break
    return st, list(todo)
