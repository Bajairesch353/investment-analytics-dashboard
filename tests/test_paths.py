import importlib

import src.paths as paths


def test_default_data_dir_is_project_data(monkeypatch):
    monkeypatch.delenv("INVESTMENT_DATA_DIR", raising=False)
    reloaded = importlib.reload(paths)
    assert reloaded.DATA_DIR == reloaded.PROJECT_ROOT / "data"
    assert reloaded.PROCESSED_PATH == reloaded.DATA_DIR / "processed"
    assert reloaded.MANUAL_PATH == reloaded.DATA_DIR / "manual"
    assert reloaded.RAW_PATH == reloaded.DATA_DIR / "raw"


def test_env_var_overrides_data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("INVESTMENT_DATA_DIR", str(tmp_path))
    reloaded = importlib.reload(paths)
    assert reloaded.DATA_DIR == tmp_path.resolve()
    assert reloaded.PROCESSED_PATH == tmp_path.resolve() / "processed"
    monkeypatch.delenv("INVESTMENT_DATA_DIR")
    importlib.reload(paths)
