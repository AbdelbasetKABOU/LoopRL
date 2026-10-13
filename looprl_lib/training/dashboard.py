"""
Streamlit application to visualize training.

To run:
    streamlit run looprl_lib/training/dashboard.py -- [session_dir]

Application specification:
    - We can visualize any iteration (pretraining/0/1/...)

References:
    - Streamlit API: https://docs.streamlit.io/library/api-reference
"""

import argparse
import datetime
import json
import os
import random
from os.path import join

import altair as alt  # type: ignore
import pandas as pd  # type: ignore
import streamlit as st
from looprl import AgentSpec
from looprl_lib.disk_recorder import indexed_elements_in_dir
from looprl_lib.events import EventsSpec
from looprl_lib.net_util import EPOCHS_STATS_FILE, STEPS_STATS_FILE
from looprl_lib.samples import load_sample
from looprl_lib.training.agent import Agent, solver, teacher
from looprl_lib.training.session import (LOG_FILE, PARAMS_DIFF_FILE,
                                         PROBLEMS_DIR, SAMPLES_DIR, STATS_FILE,
                                         TIME_FILE, TRAIN_DATA_DIR,
                                         TRAINING_DIR, VALIDATION_DATA_DIR,
                                         cur_session_dir, file, read_params,
                                         set_cur_session_dir, subdir)

EXP_PARAMS_TAB = "Experiment Parameters"
LOG_TAB = "Training Log"
AGENTS_STATS_TAB = "Agent Stats"
PERF_CURVES_TAB = "Agent Performance Curves"
PROBLEMS_TAB = "Generated Problems"
SAMPLES_TAB = "Agent Samples"
NET_TRAINING_TAB = "Network Training"
LOSSES_TAB = "Validation Loss Evolution"
TIME_TAB = "Stage Times"


def run(session_dir: str) -> None:
    st.set_page_config(layout="wide")
    st.title("Looprl Training Dashboard")
    set_cur_session_dir(session_dir)
    with st.sidebar:
        st.title("Navigation")
        tab = st.radio("Select Tab:", [
            EXP_PARAMS_TAB, LOG_TAB, PERF_CURVES_TAB, AGENTS_STATS_TAB,
            PROBLEMS_TAB, SAMPLES_TAB, NET_TRAINING_TAB, LOSSES_TAB, TIME_TAB])
    if tab == EXP_PARAMS_TAB:
        viz_exp_params()
    if tab == LOG_TAB:
        viz_log()
    if tab == PERF_CURVES_TAB:
        viz_perf_curves()
    if tab == AGENTS_STATS_TAB:
        viz_agent_stats()
    if tab == PROBLEMS_TAB:
        viz_problems()
    if tab == SAMPLES_TAB:
        viz_samples()
    if tab == NET_TRAINING_TAB:
        viz_grad_updates()
    if tab == LOSSES_TAB:
        viz_losses()
    if tab == TIME_TAB:
        viz_times()


def format_time(seconds: float) -> str:
    delta = datetime.timedelta(seconds=round(seconds))
    return str(delta)


def viz_times() -> None:
    st.header(TIME_TAB)
    file = join(cur_session_dir(), TIME_FILE)
    with open(file, "r") as f:
        times = json.load(f)
    total = sum(times.values())
    records = []
    for stage_str, t in times.items():
        records.append({
            'stage': stage_str,
            'time': format_time(t)})
    st.markdown(f"**Total time**: {format_time(total)}")
    st.table(pd.DataFrame.from_records(records))


def viz_exp_params() -> None:
    params = read_params()
    with open(file(PARAMS_DIFF_FILE), 'r') as f:
        diff = json.load(f)
    st.header(EXP_PARAMS_TAB)
    if params is None:
        st.text("Parameters file not found.")
    else:
        st.text(f"Hyperparameters diff for session at: {cur_session_dir()}")
        st.text("")
        st.json(diff)
        st.text(f"Complete hyperparameters listing")
        st.text("")
        st.json(params.to_json())  #type: ignore


def viz_log() -> None:
    log_file = join(cur_session_dir(), LOG_FILE)
    st.header(LOG_TAB)
    st.text("")
    if os.path.isfile(log_file):
        with open(log_file, "r") as f:
            st.text(f.read())
    else:
        st.write("No log file found.")


def num_finished_iterations(agent: Agent) -> int:
    i = 0
    while os.path.isfile(file(agent.dir, i, TRAINING_DIR, EPOCHS_STATS_FILE)):
        i += 1
    return i


def stats_table(agent: Agent) -> pd.DataFrame:
    its = list(range(num_finished_iterations(agent)))
    stats: list[pd.DataFrame] = []
    for i in its:
        stats_file = file(agent.dir, i, TRAIN_DATA_DIR, STATS_FILE)
        stats.append(pd.read_json(stats_file))
    data = pd.concat([s.mean() for s in stats], axis=1).transpose()
    return data


def viz_agent_stats() -> None:
    st.header(AGENTS_STATS_TAB)
    agent = select_agent()
    data = stats_table(agent)
    st.table(data.transpose())


def select_agent() -> Agent:
    ps = read_params()
    assert ps is not None
    choice = st.sidebar.selectbox('Agent: ', ['teacher', 'solver'])
    if choice == 'teacher':
        return teacher(ps)
    else:
        return solver(ps)


def viz_perf_curves() -> None:
    st.header(PERF_CURVES_TAB)
    agent = select_agent()
    data = stats_table(agent)
    left_col, right_col = st.columns(2)
    with left_col:
        st.subheader("Average Rewards")
        st.plotly_chart(data['rewards'].plot())
        st.subheader("Success Rate")
        st.plotly_chart(data['success'].plot())
    with right_col:
        st.subheader("Average Policy Target Entropy")
        st.plotly_chart(data['target-entropy'].plot())
        st.subheader("Average Trace Length")
        st.plotly_chart(data['trace-length'].plot())


def viz_samples() -> None:
    st.header(SAMPLES_TAB)
    agent = select_agent()
    it_num = select_iter_num(agent)
    samples_dir = subdir(agent.dir, it_num, VALIDATION_DATA_DIR, SAMPLES_DIR)
    num_samples = len(indexed_elements_in_dir(samples_dir))
    sample_id = int(st.sidebar.number_input(
        "Sample Id:", min_value=0, max_value=num_samples-1, step=1))
    if st.sidebar.button('Random'):
        sample_id = random.randint(0, num_samples-1)
    sample_file = join(samples_dir, str(sample_id))
    sample = load_sample(agent.unserialize, sample_file)
    st.write(f"Showing sample {sample_id}")
    left_col, right_col = st.columns(2)
    with left_col:
        st.subheader("Probe")
        probe_txt = str(sample.probe)
        st.code(probe_txt, language="txt")
    with right_col:
        st.subheader("Actions")
        if sample.actions:
            records = [
                {'action': str(a), 'target': p}
                for a, p in zip(sample.actions, sample.policy_target)]
            records.sort(reverse=True, key=lambda r: r['target'])  #type: ignore
            st.table(pd.DataFrame.from_records(records))
        else:
            st.write("No actions.")
    left_col, right_col = st.columns(2)
    espec = EventsSpec(agent.spec)
    with left_col:
        st.subheader(f"Outcome predictions")
        records = [
            {'outcome': name, 'probability': sample.value_target[outcome]}
            for outcome, name in enumerate(agent.spec['outcome_names'])]
        st.table(pd.DataFrame.from_records(records))
    with right_col:
        st.subheader(f"Event predictions")
        def num_events(e):
            offset = espec.event_offsets[e]
            m = agent.spec['event_max_occurences'][e]
            for i in range(m+1):
                if sample.value_target[offset+i] != 0:
                    return i
            assert False
        records = [
            {'event': name, 'num': num_events(e)}
            for e, name in enumerate(agent.spec["event_names"])]
        st.table(pd.DataFrame.from_records(records))


def viz_grad_updates() -> None:
    st.header(NET_TRAINING_TAB)
    agent = select_agent()
    it_num = select_iter_num(agent)
    training_dir = subdir(agent.dir, it_num, TRAINING_DIR)
    show_settings_header()
    alpha = select_smoothing_alpha()
    viz_step_statistics(training_dir, alpha)
    viz_epochs_statistics(training_dir)


def viz_step_statistics(training_dir: str, alpha: float) -> None:
    st.subheader("Step Statistics")
    steps_file = join(training_dir, STEPS_STATS_FILE)
    data = pd.read_json(steps_file, lines=True)
    data['step'] = list(range(len(data.index)))
    left_column, right_column = st.columns([3, 2])
    with left_column:
        viz_loss_components(alpha, data)
    with right_column:
        viz_learning_rate(data)


def viz_loss_components(alpha: float, data: pd.DataFrame) -> None:
    loss_cols = ['step'] + [
        c for c in data.columns if c.endswith("loss") or c.endswith('dist')]
    loss_chart = alt.Chart(
        data[loss_cols]
            .ewm(alpha=alpha)
            .mean().melt('step', var_name='comp'),
        title="Train Loss").mark_line().encode(
            x=alt.X('step', title=None, axis=alt.Axis(
                tickCount=5)),
            y=alt.Y('value', title=None),
            tooltip=['step', 'comp', 'value'],
            color=alt.Color('comp', legend=alt.Legend(
                orient='top-right')))
    # TODO: add vertical rules:
    # https://github.com/altair-viz/altair/issues/2379
    st.altair_chart(
        loss_chart.interactive(),
        use_container_width=True)


def viz_learning_rate(data: pd.DataFrame):
    st.altair_chart(
        alt.Chart(
            data[['step', 'lr']],
            title="Learning rate").mark_line().encode(
                x=alt.X('step', title=None),
                y=alt.X('lr',
                    title=None,
                    axis=alt.Axis(format='.1e')),
                tooltip=['step', 'lr']),
        use_container_width=True)


def viz_epochs_statistics(training_dir: str) -> None:
    st.subheader("Epoch statistics")
    epochs_file = join(training_dir, EPOCHS_STATS_FILE)
    data = pd.read_json(epochs_file, lines=True)
    viz_validation_loss(data)


def viz_validation_loss(data: pd.DataFrame) -> None:
    data['step'] = list(range(len(data.index)))
    loss_cols = ['step'] + [c for c in data.columns if is_loss_column(c)]
    loss_chart = alt.Chart(
        data[loss_cols].melt('step', var_name='comp'),
        title="Train Loss").mark_line().encode(
            x=alt.X('step', title=None),
            y=alt.Y('value', title=None),
            tooltip=['step', 'comp', 'value'],
            color=alt.Color('comp', legend=alt.Legend(
                orient='top-right')))
    st.altair_chart(
        loss_chart.interactive(),
        use_container_width=True)


def is_loss_column(c: str) -> bool:
    return c.endswith("loss") or c.endswith("dist")


def viz_losses() -> None:
    st.header(LOSSES_TAB)
    st.subheader("Teacher")
    ps = read_params()
    assert ps is not None
    viz_losses_for(teacher(ps))


def viz_losses_for(agent: Agent) -> None:
    n = num_finished_iterations(agent)
    if n == 0:
        st.write("No iteration has been completed yet.")
        return
    records: list[dict] = []
    for i in range(n):
        stats_file = file(agent.dir, i, TRAINING_DIR, EPOCHS_STATS_FILE)
        with open(stats_file, "r") as f:
            for l in f.readlines():
                records.append(json.loads(l))
    data = pd.DataFrame.from_records(records)
    viz_validation_loss(data)


def select_iter_num(agent: Agent) -> int:
    num_iters = agent.params.num_iters
    return int(st.sidebar.number_input("Iteration: ", 0, num_iters, step=1))


def show_settings_header():
    st.sidebar.title("Settings")


def select_smoothing_alpha():
    return 1 - st.sidebar.slider("Smoothing:",
        min_value=0.0, max_value=0.95, value=0.9)


def viz_problem(agent_spec: AgentSpec, problem: dict):
    st.write(f"Outcome: {agent_spec['outcome_names'][problem['outcome']]}")
    evs = [agent_spec['event_names'][e] for e in problem['events']]
    st.write(f"Events: {', '.join(evs)}")
    if 'spec' in problem:
        st.code(problem['spec'], language="txt")
    left_col, right_col = st.columns(2)
    with left_col:
        if 'solved' in problem:
            st.code(problem['solved'], language="txt")
        if 'nonprocessed' in problem:
            st.code(problem['nonprocessed'], language="txt")
    with right_col:
        if 'problem' in problem:
            st.code(problem['problem'], language="txt")


def viz_problems_in_dir(agent: Agent, problems_dir: str):
    num_problems = len(indexed_elements_in_dir(problems_dir))
    problem_id = int(st.sidebar.number_input(
        "Problem Id:", min_value=0, max_value=num_problems-1, step=1))
    if st.sidebar.button('Random'):
        problem_id = random.randint(0, num_problems-1)
    problem_file = join(problems_dir, str(problem_id))
    with open(problem_file, 'r') as f:
        problem = json.load(f)
    st.write(f"Showing problem {problem_id}")
    viz_problem(agent.spec, problem)


def viz_problems() -> None:
    st.header(PROBLEMS_TAB)
    agent = select_agent()
    it_num = select_iter_num(agent)
    problems_dir = file(agent.dir, it_num, VALIDATION_DATA_DIR, PROBLEMS_DIR)
    viz_problems_in_dir(agent, problems_dir)


if __name__ == '__main__':
    pd.options.plotting.backend = "plotly"
    parser = argparse.ArgumentParser(
        prog='looprl-dashboard',
        description='The Looprl Training Dashboard.')
    parser.add_argument('session', type=str)
    parser.add_argument('--problems', type=str)
    args = parser.parse_args()
    run(session_dir=args.session)
