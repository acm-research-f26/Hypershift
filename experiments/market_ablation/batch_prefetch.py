"""Bounded CPU lookahead for deterministic, read-only batch preparation."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager


@contextmanager
def prefetched_batches(schedule, batch_size, prepare, *, start=0):
    """Yield offset/origins/data in order, with at most one future batch.

    `prepare` must perform only deterministic CPU reads/NumPy operations. Device
    transfers, model execution, gradients, and RNG operations stay in the caller.
    The explicit context joins the worker on interruption or preparation errors.
    """
    if batch_size < 1 or start < 0:
        raise ValueError("Positive batch size and nonnegative resume offset required")
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="market-CPU-batch")

    def batches():
        offsets = iter(range(start, len(schedule), batch_size))
        offset = next(offsets, None)
        if offset is None:
            return
        origins = schedule[offset:offset + batch_size]
        future = pool.submit(prepare, origins)
        while True:
            prepared = future.result()
            next_offset = next(offsets, None)
            if next_offset is not None:
                following = schedule[next_offset:next_offset + batch_size]
                future = pool.submit(prepare, following)
            yield offset, origins, prepared
            if next_offset is None:
                break
            offset, origins = next_offset, following

    iterator = batches()
    try:
        yield iterator
    finally:
        iterator.close()
        pool.shutdown(wait=True, cancel_futures=True)
