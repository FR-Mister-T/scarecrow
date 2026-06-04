from importlib import resources
from pathlib import Path

import pytest

from scarecrow.model import BUNDLED_WEIGHTS_FILENAME, _bundled_weights_resource, _verify_bundled_weights, load


class TestVerifyBundledWeights:
    def test_rejects_mismatched_bytes(self, tmp_path):
        """Mismatched file bytes raise RuntimeError."""
        bad = tmp_path / "weights.pt2"
        bad.write_bytes(b"not the real weights")
        with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
            _verify_bundled_weights(str(bad))

    def test_bundled_resource_exists(self):
        """The bundled weights are available as scarecrow.data package data."""
        resource = resources.files("scarecrow.data").joinpath(BUNDLED_WEIGHTS_FILENAME)
        assert resource.is_file()

    def test_bundled_resource_matches_pinned_hash(self):
        """The committed bundled weights resource matches BUNDLED_WEIGHTS_SHA256."""
        with resources.as_file(_bundled_weights_resource()) as weights:
            _verify_bundled_weights(weights)


class TestLoad:
    def test_raises_before_torch_export_load_for_bundled_resource(self, tmp_path, monkeypatch):
        """Hash verification runs before torch.export.load for bundled weights."""
        tmp_file = tmp_path / BUNDLED_WEIGHTS_FILENAME
        tmp_file.write_bytes(b"bogus")
        calls = []
        monkeypatch.setattr("scarecrow.model._bundled_weights_resource", lambda: tmp_file)
        monkeypatch.setattr("torch.export.load", lambda *a, **k: calls.append(a))
        with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
            load(device="cuda")
        assert calls == []

    def test_skips_verification_for_custom_paths(self, tmp_path, monkeypatch):
        """Explicit custom paths skip bundled verification and reach torch.export.load."""
        tmp_file = tmp_path / BUNDLED_WEIGHTS_FILENAME
        tmp_file.write_bytes(b"arbitrary")

        class Marker(Exception):
            pass

        def fake_load(weights, *args, **kwargs):
            assert Path(weights) == tmp_file
            raise Marker

        monkeypatch.setattr("torch.export.load", fake_load)
        with pytest.raises(Marker):
            load(str(tmp_file), device="cuda")
