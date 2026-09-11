from __future__ import annotations

from los80 import run


def test_default_command_executes_pipeline_and_prints_summary(monkeypatch, capsys) -> None:
    called = []

    class FakeRuntime:
        def __init__(self, *args, **kwargs):
            pass

        def ensure(self):
            return None

    class FakePipeline:
        def __init__(self, config, database):
            called.append((config, database))

        def run(self):
            return {"completed": 2, "skipped": 1, "failed": 1, "elapsed_time": 3.25}

    monkeypatch.setattr(run, "RealESRGANRuntime", FakeRuntime)
    monkeypatch.setattr(run, "Pipeline", FakePipeline)
    monkeypatch.setattr("sys.argv", ["los80", "--config", "config.yaml"])

    run.main()

    output = capsys.readouterr().out
    assert len(called) == 1
    assert "completed: 2" in output
    assert "skipped: 1" in output
    assert "failed: 1" in output
    assert "elapsed time: 3.25s" in output
