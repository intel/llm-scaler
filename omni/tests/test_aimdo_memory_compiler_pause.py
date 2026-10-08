"""Caller pause ownership when Kitchen reuses an allocation context."""
from contextlib import contextmanager, nullcontext
import threading
from types import SimpleNamespace

import pytest
from test_aimdo_memory_compiler_adapter import load_adapter


def pause_class(monkeypatch):
    adapter = load_adapter(monkeypatch)
    events = []
    class Graphs(dict):
        def __setitem__(self, key, value):
            graph = adapter._Graph(runtime, value, None)
            graph._comfy_active = value._comfy_active
            super().__setitem__(key, graph)
    graphs = Graphs()

    @contextmanager
    def paused(graph, *, sync):
        assert graph.owner is threading.current_thread()
        events.append((graph.name, "enter", sync))
        try:
            yield
        finally:
            assert graph.owner is threading.current_thread(), "pause resumed by another thread"
            events.append((graph.name, "exit", sync))

    runtime = SimpleNamespace(owner=SimpleNamespace(paused_graph_scope=paused),
        graph=lambda: graphs.get(threading.get_ident()), original_pause=lambda sync: nullcontext())
    return lambda sync=False: adapter._Pause(runtime, sync), graphs, events


@pytest.mark.parametrize("second_active", (True, False))
def test_shared_kitchen_context_keeps_each_threads_pause(second_active, monkeypatch):
    cls, graphs, events = pause_class(monkeypatch)
    shared = cls(sync=True)
    entered_a, entered_b, exited_a = (threading.Event() for _ in range(3))
    errors = []

    def worker(name, active):
        graphs[threading.get_ident()] = SimpleNamespace(
            _comfy_active=active, _comfy_xpu_diagnostic=True,
            owner=threading.current_thread(), name=name)
        try:
            if name == "B":
                assert entered_a.wait(5)
            with shared:
                (entered_a if name == "A" else entered_b).set()
                assert (entered_b if name == "A" else exited_a).wait(5)
        except BaseException as error:
            errors.append(error)
        finally:
            if name == "A":
                exited_a.set()

    workers = [threading.Thread(target=worker, args=("A", True)),
               threading.Thread(target=worker, args=("B", second_active))]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(10)
    assert not any(worker.is_alive() for worker in workers)
    assert not errors
    expected = [("A", "enter", True)]
    if second_active:
        expected.append(("B", "enter", True))
    expected.append(("A", "exit", True))
    if second_active:
        expected.append(("B", "exit", True))
    assert events == expected


def test_shared_context_retains_nested_and_exceptional_unwind(monkeypatch):
    cls, graphs, events = pause_class(monkeypatch)
    graphs[threading.get_ident()] = SimpleNamespace(
        _comfy_active=True, _comfy_xpu_diagnostic=True,
        owner=threading.current_thread(), name="owner")
    shared = cls()
    with pytest.raises(ValueError, match="cancel"):
        with shared:
            with shared:
                raise ValueError("cancel")
    with shared:
        pass
    assert [row[1] for row in events] == ["enter", "enter", "exit", "exit", "enter", "exit"]
