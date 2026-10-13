"""
A simple CLI interface to visualize search trees
"""


import argparse
import asyncio
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional, cast

import looprl
import numpy as np
import torch
from ansi.colour.fg import boldgreen, boldred, green, red  # type: ignore
from ansi.colour.fx import bold, inverse  # type: ignore
from async_timeout import sys
from prompt_toolkit.application import Application
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.bindings.scroll import (scroll_one_line_down,
                                                        scroll_one_line_up)
from prompt_toolkit.layout import Dimension
from prompt_toolkit.layout.containers import HSplit, VSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Frame, Label

from looprl_lib.events import event_counts_dict, num_outcomes
from looprl_lib.inference import make_nonbatched_network_oracle
from looprl_lib.params import (MctsParams, NetworkParams, SolverParams,
                               TeacherParams)

from .env_wrapper import (ChoiceState, FinalState, Oracle, OutcomeType,
                          StateWrapper, init_solver, init_teacher)
from .examples import EXAMPLES, code2inv
from .explore_tree import META_FOCUS
from .mcts import GumbelOutput
from .mcts import Node as MctsNode
from .pp_table import Column, StyledStr, compose_styles, no_style, pp_table
from .prettify import highlight_focus, prettify, with_unicode_operators

MCTS_TIMEOUT = 1000
MCTS_EXTRA = 100

NUM_DISPLAYED_ACTIONS = 8

LEFT_PANELS_WIDTH = Dimension(weight=60)
RIGHT_PANELS_WIDTH = Dimension(weight=100)
TOP_PANELS_HEIGHT = Dimension(weight=50)
BOTTOM_PANELS_HEIGHT = Dimension(preferred=(NUM_DISPLAYED_ACTIONS + 4))
MIN_COL_WIDTH = None
MIN_LEFT_COL_WIDTH = 30

UNAVAILABLE_STR = "-"  # "-"
CUR_NODE_LABEL_STYLE = inverse
MCTS_TREE_HEADER_STYLE = bold
SELECTED_ACTION_STYLE = inverse
HEADER_BOT_MARGIN = 0
COLS_SEP = 3

BUFFER_TOP_PADDING = ""
ACTIONS_BUFFER_TOP_PADDING = "\n"

DEFAULT_STATUS_MSG = "Press ? for help."
SPEC_CLIPBOARD_FILE = "spec.txt"


@dataclass
class MctsInfo:
    tree: MctsNode
    num_actions_before_root: int
    gumbel_output: Optional[GumbelOutput]


@dataclass
class ExplorerApp:
    state: StateWrapper
    mcts_params: MctsParams
    show_success: Callable[[Any], str] = lambda x: str(x)

    def __post_init__(self):
        self.mcts: Optional[MctsInfo] = None
        self.actions_history: list[int] = []
        self.state_history: list[StateWrapper] = []
        self.items: list[int] = []
        self.selected_item_idx: Optional[int] = None

        self.show_encoding_mode = False
        self.show_value_pred_mode = False

        self.rng = np.random.default_rng()

        self.probe_view = FormattedTextControl(show_cursor=False)
        self.info_view = BufferControl(focusable=False)
        self.actions_view = FormattedTextControl(show_cursor=False)
        self.status_bar = Label("", style="reverse")

        self.probe_frame = Frame(
            title="Probe",
            width=LEFT_PANELS_WIDTH,
            height=TOP_PANELS_HEIGHT,
            body=Window(self.probe_view, wrap_lines=True))

        self.info_frame = Frame(
            title="Info",
            width=RIGHT_PANELS_WIDTH,
            height=TOP_PANELS_HEIGHT,
            body=Window(self.info_view, wrap_lines=True))

        self.actions_frame = Frame(
            title="Actions",
            height=BOTTOM_PANELS_HEIGHT,
            body=Window(self.actions_view, wrap_lines=True))

        container = HSplit([
            Label(" Looprl", style="reverse"),
            VSplit([self.probe_frame, self.info_frame]),
            self.actions_frame,
            self.status_bar
        ])

        kb = KeyBindings()

        @kb.add("c-c")
        @kb.add("c-q")
        def quit(event):
            event.app.exit()

        @kb.add("up")
        def prev_item(_):
            self.move_item_selection("up")
            self.refresh()

        @kb.add("down")
        def next_item(_):
            self.move_item_selection("down")
            self.refresh()

        @kb.add("s-up")
        def _(event):
            w = event.app.layout.current_window
            event.app.layout.focus(self.info_frame.body)
            scroll_one_line_up(event)
            event.app.layout.focus(w)

        @kb.add("s-down")
        def _(event):
            w = event.app.layout.current_window
            event.app.layout.focus(self.info_frame.body)
            scroll_one_line_down(event)
            event.app.layout.focus(w)

        @kb.add("enter")
        async def select_item(_):
            await self.select_item()
            self.refresh()

        @kb.add("backspace")
        async def move_to_parent_node(_):
            await self.move_to_parent_node()
            self.refresh()

        @kb.add("e")
        def toggle_show_encoding_mode(_):
            self.show_encoding_mode = not self.show_encoding_mode
            self.refresh()

        @kb.add("v")
        def toggle_show_value_pred_mode(_):
            self.show_value_pred_mode = not self.show_value_pred_mode
            self.refresh()

        @kb.add("t")
        async def init_mcts_tree(_):
            await self.create_mcts_tree()
            self.refresh()

        @kb.add("m")
        async def run_mcts(_):
            if (node := self.cur_mcts_node()) is not None:
                if not node.success:
                    await node.solve(MCTS_TIMEOUT)
                else:
                    await node.explore(MCTS_EXTRA)
                await self.update_action_items()
                self.refresh()

        @kb.add("g")
        async def run_gumbel(_):
            if (node := self.cur_mcts_node()) is not None:
                out = await node.gumbel_explore(self.rng)
                assert self.mcts is not None
                self.mcts.gumbel_output = out
                self.refresh()

        @kb.add("s")
        async def run_mcts_simulation(_):
            if (node := self.cur_mcts_node()) is not None:
                await node.explore(1)
                await self.update_action_items()
                self.refresh()

        @kb.add("r")
        async def resample(_):
            if self.actions_history:
                a = self.actions_history[-1]
                await self.move_to_parent_node()
                await self.perform_action(a)
                self.refresh()
                self.set_status_msg("Resampled.")

        @kb.add(":")
        async def copy_spec(_):
            if isinstance(self.status, ChoiceState):
                meta = self.state.probe.meta()
                if 'spec_sexp' in meta:
                    with open(SPEC_CLIPBOARD_FILE, 'w') as f:
                        f.write(meta['spec_sexp'])
                    self.set_status_msg(
                        f"Copied spec sexp in '{SPEC_CLIPBOARD_FILE}'.")
            else:
                self.set_status_msg("Unavailable spec sexp.")

        style = Style([])
        self.app: Application = Application(
            layout=Layout(container),
            key_bindings=kb,
            style=style,
            full_screen=True)

        self.app.layout.focus(self.actions_frame.body)

    def reset_items(self) -> None:
        self.selected_item_idx = None
        self.items = []

    async def update_action_items(self) -> None:
        self.reset_items()
        status = await self.state.status()
        if isinstance(status, ChoiceState):
            if (node := self.cur_mcts_node()) is not None:
                nvisits = node.children_visits()
                self.items = np.argsort(-nvisits).tolist()
            elif self.state.is_chance_node:
                probs = self.state.normalized_weights
                self.items = np.argsort(-probs).tolist()
            else:
                self.items = list(range(len(self.state.actions)))
            self.selected_item_idx = 0

    def move_item_selection(self, move: Literal['up', 'down']) -> None:
        n = len(self.items)
        cur = self.selected_item_idx
        if cur is not None:
            new = cur - 1 if move == 'up' else cur + 1
            if new >= 0 and new < n:
                self.selected_item_idx = new

    @property
    def selected_item(self) -> Optional[int]:
        if self.selected_item_idx is None:
            return None
        else:
            return self.items[self.selected_item_idx]

    async def perform_action(self, idx: int) -> None:
        if isinstance(self.status, ChoiceState):
            self.actions_history.append(idx)
            self.state_history.append(self.state)
            self.state = self.state.select(idx)
            await self.current_state_updated()

    async def select_item(self) -> None:
        if (idx := self.selected_item) is None:
            self.reset_items()
            return
        await self.perform_action(idx)

    async def move_to_parent_node(self) -> None:
        if self.actions_history:
            self.actions_history.pop()
            self.state = self.state_history.pop()
            if (self.mcts is not None and
                self.mcts.num_actions_before_root > len(self.actions_history)):
                self.mcts = None
            await self.current_state_updated()

    async def create_mcts_tree(self) -> None:
        tree = await MctsNode.make(self.state, self.mcts_params)
        num_actions_before_mcts_root = len(self.actions_history)
        self.mcts = MctsInfo(tree, num_actions_before_mcts_root, None)

    def cur_mcts_node(self) -> Optional[MctsNode]:
        # The current MCTS node is identified in the tree using
        # the actions history.
        if self.mcts is None:
            return None
        else:
            depth = self.mcts.num_actions_before_root
            assert depth <= len(self.actions_history)
            path = self.actions_history[depth:]
            return self.mcts.tree.get_node(path)

    async def current_state_updated(self) -> None:
        self.status = await self.state.status()
        if self.mcts is not None:
            self.mcts.gumbel_output = None
        await self.update_action_items()

    def set_status_msg(self, msg: str) -> None:
        self.status_bar.text = " " + msg

    def refresh_probe_view(self) -> None:
        lines = []
        if isinstance(self.status, FinalState):
            outcome = self.status.outcome_type
            if outcome == OutcomeType.SUCCESS:
                lines += [boldgreen("success")]
                lines += [prettify(self.show_success(self.state.success_value))]
            elif outcome == OutcomeType.FAILURE:
                lines += [boldred("failure")]
                lines += [self.state.failure_message]
            elif outcome == OutcomeType.EMPTY_CHOICE:
                lines += [boldred("failure")]
                lines += ["empty choice"]
            elif outcome == OutcomeType.PROOF_SIZE_LIMIT_EXCEEDED:
                lines += [boldred("failure")]
                lines += ["proof size limit exceeded"]
            elif outcome == OutcomeType.PROBE_SIZE_LIMIT_EXCEEDED:
                lines += [boldred("failure")]
                lines += ["probe size limit exceeded"]
            elif outcome == OutcomeType.ACTION_SIZE_LIMIT_EXCEEDED:
                lines += [boldred("failure")]
                lines += ["action size limit exceeded"]
            else:
                assert False
        elif isinstance(self.status, ChoiceState):
            probe = prettify(str(self.state.probe))
            meta = self.state.probe.meta()
            if META_FOCUS in meta:
                probe = highlight_focus(probe, meta[META_FOCUS], style=red)
            lines += [probe]
        else:
            assert False
        self.probe_view.text = ANSI(BUFFER_TOP_PADDING + "\n\n".join(lines))

    def refresh_info_view(self) -> None:
        lines: list[str] = []
        def statline(k, v, label_pad=18):
            nonlocal lines
            lines += [f"{k+':':{label_pad}s} {v}"]
        if isinstance(self.status, FinalState):
            spec = self.state.params.espec.agent_spec
            outcome = spec['outcome_names'][self.status.outcome_code]
            lines += [f"outcome: {outcome}"]
            lines += [""]
            statline("reward", f"{self.state.final_reward:.2f}")
        if isinstance(self.status, ChoiceState):
            if self.show_encoding_mode:
                lines += [self.state.probe.graph()]
                if self.selected_item is not None:
                    lines += [self.state.actions[self.selected_item].graph()]
            elif self.show_value_pred_mode and not self.state.is_chance_node:
                lines += ["outcome-predictions:"]
                pred = self.status.oracle_output.events
                espec = self.state.params.espec
                aspec = espec.agent_spec
                pad = 4 + max(
                    [len(s) for s in aspec['outcome_names']] +
                    [len(s) for s in aspec['event_names']])
                for (i, outcome) in enumerate(aspec['outcome_names']):
                    statline("- " + outcome, f"{pred[i]:.2f}", pad)
                lines += ["", "event-predictions:"]
                for (i, event) in enumerate(aspec['event_names']):
                    offset = espec.event_offsets[i]
                    nout = num_outcomes(aspec)
                    m = aspec['event_max_occurences'][i]
                    details = ", ".join([
                        f"{pred[offset+nout+j]:.2f}"
                        for j in range(m + 1)])
                    statline("- " + event, details, pad)
            else:
                meta = self.state.probe.meta()
                meta.pop(META_FOCUS, None)
                meta.pop('spec_sexp', None)
                for k, v in meta.items():
                    lines += [k + ":\n" + with_unicode_operators(v), ""]
                # Statistic on probe size
                statline("probe-size", self.state.probe_size)
                statline("max-action-size", self.status.max_action_size)
                # Statistics on number of steps
                statline("nsteps", self.state.nsteps)
                # Statistics on prior value and details
                if not self.state.is_chance_node:
                    statline("prior-value",
                             f"{self.status.predicted_value:.2f}")
                if (node := self.cur_mcts_node()) is not None:
                    lines += [""]
                    statline("value", f"{node.value:.2f}")
                    statline("num-visits", node.num_visits)
                    if node.success_value is not None:
                        statline("success-value", f"{node.success_value:.2f}")
        if self.state.events:
            spec = self.state.params.espec.agent_spec
            counts = event_counts_dict(self.state.events, spec)
            lines += ["", "events:"]
            for e, c in counts.items():
                if c != 0:
                    lines += ["- " + e + (f" (x{int(c)})" if c > 1 else "")]

        if self.state.messages:
            lines += ["", "Messages:", ""]
            for msg in reversed(self.state.messages):
                lines += [msg]
        self.info_view.buffer.text = BUFFER_TOP_PADDING + "\n".join(lines)

    def refresh_actions_view(self) -> None:
        content = ""
        if isinstance(self.status, ChoiceState):
            action_labels: list[StyledStr] = []
            choices = self.state.actions
            # We determine what items to display (to emulate some scrolling)
            first_displayed = (
                0 if self.selected_item_idx is None
                else max(0, self.selected_item_idx - NUM_DISPLAYED_ACTIONS + 1))
            displayed_rows = range(
                first_displayed,
                min(first_displayed+NUM_DISPLAYED_ACTIONS, len(self.items)))
            # We render the action labels
            for i in self.items:
                # Is it a success node?
                style = no_style
                if ((node := self.cur_mcts_node()) is not None and
                    (child := node.children[i]) is not None and
                        child.success):
                    style = green
                if self.selected_item == i:
                    style = compose_styles(SELECTED_ACTION_STYLE, style)
                action_labels.append((
                    with_unicode_operators(str(choices[i])), style))
            cols = [Column("", action_labels, min_width=MIN_LEFT_COL_WIDTH)]
            if (bias := self.state.bias_distribution) is not None:
                    cols += [numcol("bias",
                                [f"{bias[i]:.2f}" for i in self.items])]
            # Choice nodes
            if not self.state.is_chance_node:
                prior = self.status.oracle_output.policy
                cols += [numcol("prior",
                                [f"{prior[i]:.2f}" for i in self.items])]
            # MCTS stats: q-values, num-visits, uct-scores
            if (node := self.cur_mcts_node()) is not None:
                qvals = node.completed_qvalues(fpu_red=True)
                nvisits = node.children_visits()
                targets = node.target_policy(fpu_red=True)
                cols += [
                    numcol("visits", [f"{nvisits[i]}" for i in self.items]),
                    numcol("qvalue", [f"{qvals[i]:.2f}" for i in self.items]),
                    numcol("target", [f"{targets[i]:.2f}" for i in self.items])]
                assert self.mcts is not None
                if (g := self.mcts.gumbel_output) is not None:
                    cols += [
                        numcol("gumbel",
                            [f"{g.gumbel_vars[i]:.2f}" for i in self.items]),
                        numcol("gnum",
                            [str(g.gumbel_visits[i]) for i in self.items])]
            content = pp_table(cols,
                               header_styling=MCTS_TREE_HEADER_STYLE,
                               header_bot_margin=HEADER_BOT_MARGIN,
                               displayed_rows=displayed_rows,
                               cols_sep=COLS_SEP)
        self.actions_view.text = ANSI(ACTIONS_BUFFER_TOP_PADDING + content)

    def refresh(self) -> None:
        self.refresh_probe_view()
        self.refresh_info_view()
        self.refresh_actions_view()
        self.set_status_msg(DEFAULT_STATUS_MSG)

    async def run(self) -> None:
        # avoiding a cursor from polluting the main label
        # see https://github.com/prompt-toolkit/python-prompt-toolkit/issues/827
        # self.app.output.show_cursor = lambda:None
        await self.current_state_updated()
        self.refresh()
        await self.app.run_async()


def numcol(header: str, data: list[StyledStr]):
    return Column(header, data, align_right=True, min_width=MIN_COL_WIDTH)


def load_oracle(
    net_file: Optional[str],
    net_params: NetworkParams,
    tconf: looprl.TensorizerConfig,
    agent_spec: looprl.AgentSpec
) -> Optional[Oracle]:
    if net_file is None:
        return None
    else:
        weights = torch.load(net_file)
        return make_nonbatched_network_oracle(
            weights, net_params, tconf, agent_spec)


async def explore_solver(prog: looprl.Prog, net_file: Optional[str]) -> None:
    params = SolverParams()
    tconf = params.agent.encoding.tensorizer_config
    agent_spec = looprl.solver_spec
    oracle = load_oracle(net_file, params.agent.network, tconf, agent_spec)
    state = init_solver(prog, params.agent, oracle, log_messages=True)
    app = ExplorerApp(state, params.agent.mcts)
    await app.run()


def show_teacher_success(val: Any) -> str:
    res = cast(looprl.TeacherResult, val)
    return str(res["nonprocessed"]) + "\n\n" + str(res["problem"])


async def explore_teacher(
    net_file: Optional[str],
    spec_sexp: Optional[str] = None
) -> None:
    params = TeacherParams()
    tconf = params.agent.encoding.tensorizer_config
    agent_spec = looprl.teacher_spec
    oracle = load_oracle(net_file, params.agent.network, tconf, agent_spec)
    rng = looprl.CamlRng()
    state = init_teacher(
        params.agent, rng, oracle, log_messages=True, spec_sexp=spec_sexp)
    app = ExplorerApp(state, params.agent.mcts, show_teacher_success)
    await app.run()


async def main() -> None:
    parser = argparse.ArgumentParser(
        prog='Looprl',
        description='The Looprl CLI.')
    parser.add_argument('--teacher', action='store_true')
    parser.add_argument('--teacher-with', type=str)
    parser.add_argument('--solver', type=str)
    parser.add_argument('--net', type=str)
    args = parser.parse_args()
    if args.teacher_with is not None:
        with open(args.teacher_with, 'r') as f:
            spec = f.read()
            await explore_teacher(args.net, spec)
    if args.teacher:
        await explore_teacher(args.net)
    elif args.solver is not None:
        if args.solver in EXAMPLES:
            prog = looprl.Prog(EXAMPLES[args.solver])
        else:
            try:
                prog = code2inv(int(args.solver))
            except ValueError:
                sys.exit(f"Unknown example: {args.solver}")
        await explore_solver(prog, args.net)


if __name__ == '__main__':
    asyncio.run(main())
