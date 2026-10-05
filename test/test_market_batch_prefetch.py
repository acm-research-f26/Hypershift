from threading import Event, enumerate as threads
import numpy as np
import pytest
from experiments.market_ablation.batch_prefetch import prefetched_batches


@pytest.mark.parametrize("start", [0, 3, 5, 12])
def test_preserves_batch_order_tail_and_resume_offset(start):
    schedule = np.arange(12, dtype=np.int64)[::-1]
    original = schedule.copy()
    state = np.random.get_state()
    with prefetched_batches(schedule, 5, lambda origins: origins * 2, start=start) as batches:
        actual = list(batches)
    expected = list(range(start, len(schedule), 5))
    assert [offset for offset, _, _ in actual] == expected
    for offset, origins, prepared in actual:
        assert np.array_equal(origins, schedule[offset:offset + 5])
        assert np.array_equal(prepared, origins * 2)
    assert np.array_equal(schedule, original)
    after = np.random.get_state()
    assert state[0] == after[0] and np.array_equal(state[1], after[1]) and state[2:] == after[2:]


def test_bounds_lookahead_and_joins_on_consumer_interruption():
    started, release = Event(), Event()
    prepared = []
    def prepare(origins):
        prepared.append(int(origins[0]))
        if origins[0] == 5:
            started.set()
            assert release.wait(5)
        return origins.copy()
    with pytest.raises(RuntimeError, match="interrupted"):
        with prefetched_batches(np.arange(20), 5, prepare) as batches:
            assert next(batches)[0] == 0
            assert started.wait(5)
            assert prepared == [0, 5]
            release.set()
            raise RuntimeError("interrupted")
    assert not any(t.name.startswith("market-CPU-batch") for t in threads())
    assert prepared == [0, 5]


def test_propagates_preparation_errors_and_joins_worker():
    prepared = []
    def prepare(origins):
        prepared.append(int(origins[0]))
        if origins[0] == 5:
            raise ValueError("invalid historical batch")
        return origins.copy()
    with pytest.raises(ValueError, match="invalid historical batch"):
        with prefetched_batches(np.arange(20), 5, prepare) as batches:
            list(batches)
    assert prepared == [0, 5]
    assert not any(t.name.startswith("market-CPU-batch") for t in threads())


def test_handles_empty_schedule_without_preparing_values():
    def prepare(origins):
        pytest.fail("empty schedule must not prepare a batch")
    with prefetched_batches([], 4, prepare) as batches:
        assert list(batches) == []


@pytest.mark.parametrize("batch_size,start", [(0, 0), (4, -1)])
def test_rejects_invalid_batch_or_resume_sizes(batch_size, start):
    with pytest.raises(ValueError):
        with prefetched_batches(np.arange(4), batch_size, lambda x: x, start=start):
            pass
