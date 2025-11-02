"""
General utility functions for the darksun package.
"""

from typing import Any, Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime
import time

import numpy as np

__all__ = [
    "timer", "benchmark_func", "savefig_to",
]


@contextmanager
def timer(name: str) -> Generator[None, Any, None]:
    """
    Timer context manager.

    Args:
        name (str): Name of the task the timer is handling.
    
    Raises:
        Exception: Encoutered exception within the algorithm (if any).
    """
    def handle_time(start: float, end: float) -> str:
        """Handles output time format."""
        time_interval = end - start
        h, rem = divmod(time_interval, 3600)
        m, s = divmod(rem, 60)
        return f"{int(h):02}h:{int(m):02}m:{s:.3f}s"

    start_time = time.perf_counter()
    error = False
    print(f"    # Starting '{name}' at {datetime.now():%H:%M:%S}.")
    try:
        yield
    except Exception as e:
        error = type(e).__name__
        raise
    finally:
        end_time = time.perf_counter()
        elapsed_time = handle_time(start_time, end_time)
        mess = f"    # Finished '{name}' in {elapsed_time}"
        print(mess + ".\n" if not error else mess + f" with {error}.\n")


def benchmark_func(
    func: Callable[[Any], Any],
    *args: Any,
    iterations: int = 500,
) -> tuple[float, float, Any]:
    """
    Benchmarks input `func` by running it for a specified number of times.
    The benchmark is performed by first calling the function to account
    for JIT compilation, caching and first-call effects (not included in
    the final performance time computation).

    Args:
        func (Callable[[Any], Any]):
            Function to benchmark.
        args (Any):
            Input `func` arguments.
        iterations (int, optional (default=`500`)):
            Number of call repetitions.
    
    Returns:
        output (tuple[float, float, Any]):
            - (float): Benchmarking repetitions averaged time.
            - (float): Averaged time error.
            - (Any): Input `func` results.
    """
    func(*args)
    result = None
    rep_time = []

    for _ in range(iterations):
        start_time = time.perf_counter()
        result = func(*args)
        end_time = time.perf_counter()

        delta = end_time - start_time
        rep_time.append(delta)

    rep_time_ = np.array(rep_time)
    average, error = np.mean(rep_time_), np.std(rep_time_)

    return average, error, result


def savefig_to(
    figpath: str | Path,
    name: str,
    frmt: str = 'png',
    overwrite: bool = False,
) -> str | Path | None:
    """
    Creates the filepath to save a Figure with the chosen format.
    The func checks if the image file already exists. If so, and
    `overwrite` is False, `None` is returned to avoid overwriting.

    Args:
        figpath (str | Path):
            Directory path to save the image file, e.g., `'../data'`.
        name (str):
            Name of the image file, e.g., `'img_data`.
        frmt (str, optional (default=`'png'`)):
            Format with which the image will be saved.
        overwrite (bool, optional (default=`False`)):
            If `True`, the already saved image will be overwrited.

    Returns:
        output (str | Path | None):
            Path for saving the image file. If the file is already
            present and `overwrite` is False, `None` is returned.
    """
    def is_img(filepath: Path) -> bool:
        """Checks if a file has been already saved."""
        return filepath.is_file()
    
    filepath = f'{figpath}/{name}.{frmt}'
    if is_img(Path(filepath)) and not overwrite:
        return None
    return filepath


# end