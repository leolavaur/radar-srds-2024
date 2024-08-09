from typing import Generator, Iterable, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from trustfids.utils.log import logger

# def flatten(lst: List[Iterable]) -> Generator:
#     for el in lst:
#         if isinstance(el, Tuple):
#             yield from flatten(el)
#         else:
#             yield el


def unflatten(lst: List, depth: int = 0) -> List[Tuple]:
    """Unflatten a list.

    This function converts a list of length `n` to a list of length `n / (depth**2)`,
    where the previous elements are grouped together in tuples of two elements. The
    depth parameter controls how many levels of nesting are created.
        * If `depth` is 0, the list is returned as is.
        * If `depth` is 1, the list is unflattened to a list of tuples of two elements.
        * If `depth` is 2, the list is unflattened to a list of tuples of two tuples of
            two elements, and so on.

    Example:
    ```
    >>> lst = [1, 2, 3, 4, 5, 6, 7, 8]
    >>> unflatten(lst, depth=2)
    [((1, 2), (3, 4)), ((5, 6), (7, 8))]
    ```

    Args:
        lst (List): List to unflatten.
        depth (int, optional): Depth of nesting. Defaults to 0.

    Returns:
        List: Unflattened list.
    """
    assert (
        depth == 0 or len(lst) % (2**depth) == 0
    ), "Length of list must be divisible by `depth**2`."

    if depth == 0:
        return lst
    elif depth == 1:
        return list(zip(lst[::2], lst[1::2]))
    else:
        return [
            tuple(unflatten(sub_lst, depth - 1))
            for sub_lst in zip(*[iter(lst)] * (2**depth))
        ]
