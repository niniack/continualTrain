from dataclasses import replace
from typing import Callable, Optional, Tuple

import numpy as np
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.compiler import Compiler
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.state import State
from numpy.random import rand


class RandomHorizontalFlipSeeded(Operation):
    """Flip the image horizontally with probability flip_prob.
    Operates on raw arrays (not tensors).

    Parameters
    ----------
    flip_prob : float
        The probability with which to flip each image in the batch
        horizontally.
    """

    def __init__(self, flip_prob: float = 0.5):
        super().__init__()
        self.flip_prob = flip_prob

    def generate_code(self) -> Callable:
        my_range = Compiler.get_iterator()
        flip_prob = self.flip_prob

        def flip(images, dst, indices):
            np.random.seed(indices[-1])
            should_flip = rand(images.shape[0]) < flip_prob
            print(should_flip)
            for i in my_range(images.shape[0]):
                if should_flip[i]:
                    dst[i] = images[i, :, ::-1]
                else:
                    dst[i] = images[i]

            return dst

        flip.is_parallel = True
        flip.with_indices = True
        return flip

    def declare_state_and_memory(
        self, previous_state: State
    ) -> Tuple[State, Optional[AllocationQuery]]:
        return (
            replace(previous_state, jit_mode=True),
            AllocationQuery(previous_state.shape, previous_state.dtype),
        )
