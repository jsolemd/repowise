"""Destructive writers require verified native ownership and release only their lock."""

import json
from unittest.mock import Mock

import pytest

from repowise.core import update_lock


def test_strict_lock_releases_on_failure_and_excludes_another_writer(tmp_path):
    with (
        pytest.raises(ValueError, match="body failed"),
        update_lock.strict_update_lock(tmp_path, "head"),
    ):
        assert update_lock.read_update_lock(tmp_path)["target_commit"] == "head"
        with (
            pytest.raises(update_lock.UpdateLockUnavailableError, match="already running"),
            update_lock.strict_update_lock(tmp_path),
        ):
            pytest.fail("second writer admitted")
        assert update_lock.read_update_lock(tmp_path) is not None
        raise ValueError("body failed")
    assert update_lock.read_update_lock(tmp_path) is None


@pytest.mark.parametrize("changed", ["removed", "replaced"])
def test_strict_lock_does_not_release_a_missing_or_replaced_lock(tmp_path, monkeypatch, changed):
    release = Mock(wraps=update_lock.release_update_lock)
    monkeypatch.setattr(update_lock, "release_update_lock", release)
    with update_lock.strict_update_lock(tmp_path):
        path = update_lock.update_lock_path(tmp_path)
        if changed == "removed":
            path.unlink()
        else:
            replacement = update_lock.read_update_lock(tmp_path)
            replacement["target_commit"] = "other-operation"
            path.write_text(json.dumps(replacement))
    release.assert_not_called()
    if changed == "replaced":
        assert update_lock.read_update_lock(tmp_path) == replacement


@pytest.mark.parametrize("failure", ["write", "readback", "pid", "token"])
def test_strict_lock_refuses_unverified_ownership(tmp_path, monkeypatch, failure):
    release = Mock()
    monkeypatch.setattr(update_lock, "release_update_lock", release)
    if failure == "write":
        monkeypatch.setattr(update_lock.os, "link", Mock(side_effect=OSError("read-only")))
    else:
        def invalid_readback(_path):
            if failure == "readback":
                return None
            return {"pid": -1 if failure == "pid" else update_lock.os.getpid(),
                    "pid_create_token": "not-this-process"}
        monkeypatch.setattr(update_lock, "read_update_lock", invalid_readback)
    with (
        pytest.raises(update_lock.UpdateLockUnavailableError, match="could not be verified"),
        update_lock.strict_update_lock(tmp_path),
    ):
        pytest.fail("unverified writer admitted")
    release.assert_not_called()
