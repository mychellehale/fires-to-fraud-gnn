import logging
from functools import wraps
from typing import Any, Callable, Generator, TypeVar
from tqdm import tqdm

T = TypeVar("T")

def track_progress(desc: str) -> Callable[[Callable[..., Generator[T, None, None]]], Callable[..., Generator[T, None, None]]]:
    """
    A general-purpose decorator to add a tqdm progress bar to any generator.
    """
    def decorator(func: Callable[..., Generator[T, None, None]]) -> Callable[..., Generator[T, None, None]]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Generator[T, None, None]:
            gen = func(*args, **kwargs)
            # The 'unit="day"' can be made a variable if you want to be fancy,
            # but 'item' or 'step' is a safe default for a general util.
            pbar = tqdm(desc=desc, unit="step")
            try:
                for item in gen:
                    yield item
                    pbar.update(1)
            finally:
                pbar.close()
        return wrapper
    return decorator