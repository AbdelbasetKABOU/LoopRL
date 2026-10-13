"""
Using random walks to test state properties
such as serialization consistency.
"""

import random
from typing import Callable, Iterable

import looprl
import tqdm
from looprl import SearchTree, unserialize_solver, unserialize_teacher
from looprl_lib.examples import code2inv


def test_serialize() -> None:
    st = looprl.init_teacher(looprl.CamlRng())
    st = st.select(0)
    st_sexp = st.serialize()
    st_bis = looprl.unserialize_teacher(st_sexp)
    st_bis_sexp = st_bis.serialize()
    assert st_sexp == st_bis_sexp


def random_walk(
    st: looprl.SearchTree,
    on_choice: Callable[[looprl.SearchTree], None],
    max_depth=60,
) -> Iterable[SearchTree]:
    depth = 0
    while True:
        depth += 1
        if depth > max_depth:
            break
        if st.is_event() or st.is_message():
            st = st.next()
        elif st.is_success() or st.is_failure():
            break
        elif st.is_choice():
            yield st
            n = len(st.choices())
            on_choice(st)
            if n == 0:
                break
            st = st.select(random.randint(0, n-1))
        else:
            assert False


def test_serialization_consistency(st: looprl.SearchTree, unserialize):
    st_sexp = st.serialize()
    st_bis = unserialize(st_sexp)
    st_bis_sexp = st_bis.serialize()
    assert st_sexp == st_bis_sexp


def random_walks(
    init,
    nsteps: int = 100_000,
    max_depth: int = 60,
    unserialize = None,
    on_choice: Callable[[looprl.SearchTree], None] = lambda c: None,
):
    i = 0
    progress = tqdm.tqdm(total=nsteps)
    while i < nsteps:
        for st in random_walk(init(), on_choice, max_depth):
            if unserialize is not None:
                test_serialization_consistency(st, unserialize)
            i += 1
            progress.update()


def random_code2inv() -> SearchTree:
    prog = code2inv(random.randint(1, 100))
    return looprl.init_solver(prog)


def init_teacher() -> SearchTree:
    return looprl.init_teacher(looprl.CamlRng())


def action_size_monitor():
    max_size = 0
    def on_choice(state: looprl.SearchTree):
        nonlocal max_size
        assert state.is_choice()
        n = len(state.choices())
        if n > max_size:
            max_size = n
            print(n)
    return on_choice


def no_monitor():
    return lambda state: None


def main(monitor_action_size: bool = False):
    test_serialize()
    on_choice = action_size_monitor if monitor_action_size else no_monitor
    print("Running random walks on the solver...")
    random_walks(
        random_code2inv,
        on_choice=on_choice(),
        max_depth=12, unserialize=unserialize_solver, nsteps=100_000)
    print("Running random walks on the teacher...")
    random_walks(
        init_teacher,
        on_choice=on_choice(),
        unserialize=unserialize_teacher, nsteps=600_000)


if __name__ == '__main__':
    main()
