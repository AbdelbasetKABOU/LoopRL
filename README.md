# LoopRL

Looprl is an experimental theorem prover that provides facilities for writing proof search strategies in the form of nondeterministic programs that are then refined through AlphaZero-style reinforcement learning. It contains the following features:

- **A domain specific language for writing strategies** embedded in OCaml,  with first-class support for neural-guided nondeterministic programming. Strategies are automatically compiled into reinforcement learning environments that can be explored in Python.
- **A graphical interface** that can be used to interact with strategies manually while inspecting the neural network's predictions and the behavior of MCTS.
- **A library of utility functions for writing program verification strategies** specifically. This library includes an abduction engine for integer linear arithmetic based on Fourier-Motzkin elimination.
- **A parallel, high-performance implementation of the (Gumbel) AlphaZero algorithm** for refining search strategies. Our implementation provides support for events and unbounded, variable-size action spaces.

A teacher and a solver strategy are currently implemented for the problem of loop invariant synthesis for small imperative programs. The code for these strategies can be found in `looprl/solver.ml` and `looprl/teacher.ml`.

# Setup instructions

All commands listed in these instructions should be run in the Looprl root directory (containing this README file).

### Installing the OCaml dependencies

First of all, you should install an OCaml distribution. Looprl requires OCaml version 4.12 at least.

```sh
bash -c "sh <(curl -fsSL https://raw.githubusercontent.com/ocaml/opam/master/shell/install.sh)"
sudo apt-get install unzip bubblewrap
opam init
opam switch create 4.13+options ocaml-option-flambda
opam install dune
```

Then, you should install all OCaml dependencies. To list all dependencies that need to be installed, simply execute the following command:

```sh
dune external-lib-deps --missing @@default  # and then execute hint
```

In case some dependencies are missing, a command for installing them all will be suggested.

### Compiling the Looprl OCaml library

To build the OCaml Looprl library, just run `dune build`.

### Installing the Python dependencies

Looprl requires CPython version 3.9 at least. To install all Python dependencies, you can run:

```
pip install -r requirements.txt
```

To ensure proper GPU support, you may want to install the pytorch and pytorch-scatter dependencies separately:

```sh
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
```

### Adding Looprl to your PATH variable

To ensure that the Looprl Python libraries are found by CPython and Mypy, you must add the following to the `PYTHONPATH` and `MYPYPATH` environment variables.

```sh
export PYTHONPATH="$PYTHONPATH:$(pwd):$(pwd)/looprl/python"
export MYPYPATH="$MYPYPATH:$(pwd):$(pwd)/looprl/python"
```

### Installing the Code2Inv benchmark

Many Looprl functionalities require the Code2inv benchmark to be installed. To do so, just run the following command:

```sh
sh benchmarks/code2inv/install.sh
```

### Testing your installation

Once you've followed all the steps above, you can run the following to test your installation:

```sh
python tests/test_random_walk.py
```

You can also use Mypy to typecheck the entire codebase:

```sh
mypy looprl_lib
```

No warning should remain (using Mypy v0.950).

# Usage

## Replicating the NeurIPS 2022 experiments

To replicate all experiments from the NeurIPS 2022 paper, just run the following:

```sh
python -m looprl_lib.experiments.neurips
```

Running this script takes about 35 hours on our configuration (10-core Intel i9-10900KF processor, 64GB of RAM and a NVIDIA GeForce RTX 3080 GPU). A `neurips` folder should be created, which contains all figures and tables from the paper in `neurips/out`.


## Using the UI to explore strategies

You can use the UI to navigate the teacher and solver strategies proposed in the NeurIPS paper for loop invariant synthesis.

```
python -m looprl_lib.cli --solver 2  # specify any code2inv problem number
python -m looprl_lib.cli --teacher
python -m looprl_lib.cli --solver 3 --net net.pt  # loading a network
```

It is also possible to have the UI load a neural network for inspection. Good neural networks to try after replicating the NeurIPS experiments are to be found in `neurips/runs/0/full/solver/19/net.pt` (for the solver) and `neurips/runs/0/full/teacher/19/net.pt` (for the teacher).