import asyncio
from dataclasses import dataclass

import ray
from ray.actor import ActorHandle
from tqdm import tqdm  # type: ignore


class ProgressBarActor:

    def __init__(self):
        self.event = asyncio.Event()
        self.counter = 0
        self.delta = 0

    def tick(self) -> None:
        self.counter += 1
        self.delta += 1
        self.event.set()

    async def wait_for_update(self) -> tuple[int, int]:
        await self.event.wait()
        self.event.clear()
        saved_delta = self.delta
        self.delta = 0
        return saved_delta, self.counter


@dataclass
class ProgressBar:
    """
    # See https://docs.ray.io/en/latest/auto_examples/progress_bar.html
    """
    actor: ActorHandle
    total: int
    description: str = ""

    def print_until_done(self) -> None:
        """Blocking call.

        Do this after starting a series of remote Ray tasks, to which you've
        passed the actor handle. Each of them calls `update` on the actor.
        When the progress meter reaches 100%, this method returns.
        """
        pbar = tqdm(desc=self.description, total=self.total)
        counter = 0
        while counter < self.total:
            delta, counter = ray.get(self.actor.wait_for_update.remote())
            pbar.update(delta)
        pbar.close()
        print("")
