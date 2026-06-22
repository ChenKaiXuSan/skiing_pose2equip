import sys
import types


def test_disable_broken_torchaudio_for_transformers_marks_audio_unavailable(monkeypatch):
    import pose2equip.models.image_encoder as image_encoder

    import_utils_mod = types.SimpleNamespace(
        is_torchaudio_available=lambda: True,
    )
    utils_mod = types.SimpleNamespace(
        is_torchaudio_available=lambda: True,
    )
    transformers_mod = types.SimpleNamespace(utils=utils_mod)

    monkeypatch.setitem(sys.modules, "transformers", transformers_mod)
    monkeypatch.setitem(sys.modules, "transformers.utils", utils_mod)
    monkeypatch.setitem(sys.modules, "transformers.utils.import_utils", import_utils_mod)

    monkeypatch.delitem(sys.modules, "torchaudio", raising=False)

    def broken_import(name, *args, **kwargs):
        if name == "torchaudio":
            raise OSError("Could not load this library: _torchaudio.abi3.so")
        return original_import(name, *args, **kwargs)

    original_import = __import__
    monkeypatch.setattr("builtins.__import__", broken_import)

    assert image_encoder._disable_broken_torchaudio_for_transformers() is True
    assert import_utils_mod.is_torchaudio_available() is False
    assert utils_mod.is_torchaudio_available() is False


def test_disable_broken_torchaudio_for_transformers_leaves_working_audio(monkeypatch):
    import pose2equip.models.image_encoder as image_encoder

    import_utils_mod = types.SimpleNamespace(
        is_torchaudio_available=lambda: True,
    )
    utils_mod = types.SimpleNamespace(
        is_torchaudio_available=lambda: True,
    )
    torchaudio_mod = types.SimpleNamespace(functional=types.SimpleNamespace())

    monkeypatch.setitem(sys.modules, "transformers.utils", utils_mod)
    monkeypatch.setitem(sys.modules, "transformers.utils.import_utils", import_utils_mod)
    monkeypatch.setitem(sys.modules, "torchaudio", torchaudio_mod)

    assert image_encoder._disable_broken_torchaudio_for_transformers() is False
    assert import_utils_mod.is_torchaudio_available() is True
    assert utils_mod.is_torchaudio_available() is True
