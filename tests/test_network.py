
import looprl
import numpy as np
from looprl_lib.examples import code2inv
from looprl_lib.inference import evaluate_batch
from looprl_lib.net_util import make_network
from looprl_lib.params import EncodingParams, NetworkParams
from looprl_lib.tensors import ChoicesBatch, tensorize_choice_state


def test_network() -> None:
    # Creating some dummy tensors
    net_params = NetworkParams()
    encoding = EncodingParams()
    tconf = encoding.tensorizer_config
    st = looprl.init_solver(code2inv(2))
    choice_tensors = tensorize_choice_state(st, encoding)
    batch = ChoicesBatch.make(
        [choice_tensors] * 2,
        num_probe_toks=80,
        num_action_toks=20)
    # Creating a network
    network = make_network(net_params, tconf, looprl.solver_spec)
    network.train(mode=False)
    # Evaluating a batch with the network
    res = evaluate_batch(network, batch)
    assert np.all(np.abs(res[0].events - res[1].events) < 1e-6)
    assert np.all(np.abs(res[0].policy - res[1].policy) < 1e-6)


test_network()
