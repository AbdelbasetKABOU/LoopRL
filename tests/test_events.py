"""
Testing the reward computation system
"""

import looprl
from looprl_lib.events import (EventsSpec, event_counts, final_reward,
                               num_outcomes, value_prediction)


def pred(espec, psuccess, evs):
    pred = espec.default_pred_vec.copy()
    pred[failure] = 1 - psuccess
    pred[success] = psuccess
    nout = num_outcomes(espec.agent_spec)
    counts = event_counts(espec.agent_spec, evs)
    offsets = espec.event_offsets
    for e, c in enumerate(counts):
        if c > 0:
            offset = offsets[e]
            pred[nout+offset] = 0
            pred[nout+offset+c] = 1
    return pred


def uniform_pred(espec, event):
    pred = espec.default_pred_vec.copy()
    nout = num_outcomes(espec.agent_spec)
    m = espec.agent_spec['event_max_occurences'][event]
    off = espec.event_offsets[event]
    pred[nout+off:nout+off+m+1] = 1 / (m+1)
    return pred


def test_vpred(espec, pdiff, evs, expected):
    v = value_prediction(pred(espec, *pdiff), espec, evs)
    assert abs(v - expected) < 1e-5, f"Expected value: {expected:.3f}"


def test_uvpred(espec, event, evs, expected):
    v = value_prediction(uniform_pred(espec, event), espec, evs)
    assert abs(v - expected) < 1e-5, f"Expected value: {expected:.3f}"


def test_final(espec, evs, outcome, expected):
    v = final_reward(espec.agent_spec, evs, outcome)
    assert abs(v - expected) < 1e-5, f"Expected value: {expected:.3f}"


teacher = EventsSpec(looprl.teacher_spec)
solver = EventsSpec(looprl.solver_spec)


success = teacher.agent_spec['success_code']
failure = teacher.agent_spec['default_failure_code']
assert solver.agent_spec['success_code'] == success
assert solver.agent_spec['default_failure_code'] == failure


test_final(teacher, [], failure, -1)  # we don't care about events
test_final(solver,  [0, 1], failure, -1)
test_final(teacher, [0, 1], failure, -1)
test_final(teacher, [], success, 1)  # but we do in case of a success
test_final(teacher, [1], success, 0.5)
test_final(teacher, [1, 1], success, 0.5)  # no double counting
test_final(teacher, [1, 2], success, 0)
test_final(teacher, [1, 2, 3], success, -0.5)
test_final(solver,  [0] * 1, success, 0.8)  # respect max count value
test_final(solver,  [0] * 2, success, 0.6)
test_final(solver,  [0] * 3, success, 0.4)
test_final(solver,  [0] * 4, success, 0.2)
test_final(solver,  [0] * 5, success, 0.2)
test_final(teacher, [1, 2, 3, 4], success, -0.5)  # min_success_value
test_final(solver,  [1] * 10, success, 0)
test_final(solver,  [0, 1], success, 0.5)


test_vpred(teacher, (0.5, []), [], 0)
test_vpred(teacher, (0.5, []), [1], -0.25)
test_vpred(teacher, (0.5, [1]), [1], -0.25)
test_vpred(teacher, (0.5, [1]), [], -0.25)
test_vpred(teacher, (0.5, [1]), [], -0.25)
test_vpred(teacher, (0, [1]), [], -1)
test_vpred(teacher, (0, [1]), [2, 3], -1)
test_vpred(teacher, (1, [1]), [1, 2], 0)
test_vpred(teacher, (0.5, [1, 2, 3, 4]), [], 0.5*(-1)+0.5*(-0.5))

test_vpred(solver, (1.0, []), [0, 0], 0.6)
test_vpred(solver, (1.0, [0, 0]), [0, 0], 0.6)
# more happened than predicted in total
test_vpred(solver, (1.0, [0, 0]), [0, 0, 0], 0.4)
test_vpred(solver, (1.0, [0, 0]), [0, 0, 0], 0.4)

test_uvpred(solver, 0, [], 0.5 * (-1 + 0.6))
test_uvpred(solver, 0, [0, 0], 0.5 * (-1 + 0.4))
test_uvpred(solver, 0, [0, 0, 0, 0], 0.5 * (-1 + 0.2))
