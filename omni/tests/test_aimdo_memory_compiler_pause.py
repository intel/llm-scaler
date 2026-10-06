"""Caller pause ownership when Kitchen reuses an allocation context."""
import ast
from contextlib import contextmanager
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest


def pause_class():
    patch = Path(__file__).parents[1] / "patches/comfyui-aimdo-torch214-memory-compiler.patch"
    lines = patch.read_text().splitlines()
    start = lines.index(" class _PauseMallocGraph:")
    source = []
    for line in lines[start:]:
        if line.startswith("@@"):
            break
        if line.startswith((" ", "+")):
            source.append(line[1:])
    tree = ast.parse("\n".join(source))
    assert len(tree.body) == 1 and isinstance(tree.body[0], ast.ClassDef)
    graphs, events = {}, []

    @contextmanager
    def paused(graph, *, sync):
        assert graph.owner is threading.current_thread()
        events.append((graph.name, "enter", sync))
        try:
            yield
        finally:
            assert graph.owner is threading.current_thread(), "pause resumed by another thread"
            events.append((graph.name, "exit", sync))

    namespace = {"threading": threading, "MALLOC_GRAPHS": graphs,
                 "native_owner": SimpleNamespace(paused_graph_scope=paused)}
    exec(compile(tree, str(patch), "exec"), namespace)
    return namespace["_PauseMallocGraph"], graphs, events


@pytest.mark.parametrize("second_active", (True, False))
def test_shared_kitchen_context_keeps_each_threads_pause(second_active):
    cls, graphs, events = pause_class()
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


def test_shared_context_retains_nested_and_exceptional_unwind():
    cls, graphs, events = pause_class()
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
