import asyncio

from looprl_lib.env_wrapper import init_solver
from looprl_lib.examples import iter_code2inv
from looprl_lib.mcts import Node
from looprl_lib.params import Params


async def main() -> None:
    params = Params().solver.agent
    for i, prog in iter_code2inv():
        state = init_solver(prog, params)
        mcts = await Node.make(state, params.mcts)
        solved = await mcts.solve(timeout=5000)
        print(f"{i:3d}  {mcts.num_visits if solved else None}")


if __name__ == '__main__':
    asyncio.run(main())
