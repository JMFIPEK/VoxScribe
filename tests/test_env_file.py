"""voxscribe/env_file.py: the Settings page's HuggingFace token in `.env`."""

from voxscribe.env_file import set_env_value


def test_adds_key_to_missing_file(tmp_path):
    path = tmp_path / ".env"
    set_env_value("HF_TOKEN", "hf_abc", str(path))
    assert path.read_text(encoding="utf-8") == "HF_TOKEN=hf_abc\n"


def test_replaces_only_that_line(tmp_path):
    path = tmp_path / ".env"
    path.write_text("# my settings\nKIT_TOOLBOX_API_KEY=x\nHF_TOKEN=old\n\nOTHER=1\n", encoding="utf-8")
    set_env_value("HF_TOKEN", "hf_new", str(path))
    assert path.read_text(encoding="utf-8") == "# my settings\nKIT_TOOLBOX_API_KEY=x\nHF_TOKEN=hf_new\n\nOTHER=1\n"


def test_empty_value_removes_key(tmp_path):
    path = tmp_path / ".env"
    path.write_text("A=1\nexport HF_TOKEN = 'old'\nB=2\n", encoding="utf-8")
    set_env_value("HF_TOKEN", "", str(path))
    assert path.read_text(encoding="utf-8") == "A=1\nB=2\n"


def test_empty_value_does_not_create_file(tmp_path):
    path = tmp_path / ".env"
    set_env_value("HF_TOKEN", "", str(path))
    assert not path.exists()


def test_similar_key_names_untouched(tmp_path):
    path = tmp_path / ".env"
    path.write_text("HF_TOKEN_OLD=keep\n", encoding="utf-8")
    set_env_value("HF_TOKEN", "hf_x", str(path))
    assert path.read_text(encoding="utf-8") == "HF_TOKEN_OLD=keep\nHF_TOKEN=hf_x\n"
