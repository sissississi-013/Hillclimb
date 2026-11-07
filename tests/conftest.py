import pytest

from hillclimb import harness


@pytest.fixture(autouse=True)
def _no_repo_env(monkeypatch, tmp_path_factory):
    """harness.main() loads the repo's .env; tests must never pick up the real Fireworks key."""
    monkeypatch.setattr(harness, "ENV_FILE", tmp_path_factory.mktemp("env") / ".env")
