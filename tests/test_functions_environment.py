import os
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

from pwf import functions
from pwf import functions_environment as fenv

# make_reference.R's input files, byte for byte: no final newline, CRLF, an old Mac CR, a form feed (not a
# line end for R's readLines()), an empty file, a file the pattern leaves out and a directory it matches
ENVIRONMENT_FILES = {"a.R": b"a <- 1\nx <- 2", "Z.R": b"Z <- 1\n", "sub/c.r": b"c <- 3\r\nd <- 4\r\n",
                     "note.txt": b"not r\n", "ff.R": b"f\fg\n", "m.R": b"old\r\nmac\rline\n", "empty.R": b""}


def test_combine_files_matches_rwf(reference, tmp_path, monkeypatch, capsys):
    """R's list.files() order (its collation: Z.R after sub/c.r), readLines() lines and writeLines() line ends."""
    for name, content in ENVIRONMENT_FILES.items():
        (tmp_path / "in" / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "in" / name).write_bytes(content)
    (tmp_path / "in" / "dir.R").mkdir()
    monkeypatch.chdir(tmp_path)
    combined = {"recursive": fenv.combine_files(input_dir="in", output_file="out_rec.R", pattern=r"\.[Rr]$"),
                "flat": fenv.combine_files(input_dir="in", output_file="in/out_flat.R", pattern=r"\.[Rr]$",
                                           recursive=False)}
    assert capsys.readouterr().out.splitlines() == ["Combined 6 files into out_rec.R ",
                                                    "Combined 5 files into in/out_flat.R "]
    files = reference("combine_files_list")
    lines = reference("combine_files_lines", keep_default_na=False)
    for kind, output in [("recursive", "out_rec.R"), ("flat", "in/out_flat.R")]:
        assert combined[kind] == files.loc[files["recursive"] == kind, "file"].tolist(), kind
        raw = (tmp_path / output).read_bytes()
        assert raw.count(os.linesep.encode()) == raw.count(b"\n")  # the platform's line ends, as R
        written = raw.decode("utf-8").replace("\r\n", "\n").split("\n")[:-1]
        assert written == lines.loc[lines["recursive"] == kind, "line"].tolist(), kind


def test_environment_options_print_numbers_as_rwf(reference):
    """pandas and numpy print as R under environment_options()'s options(digits = 4, scipen = 999)."""
    printed = reference("environment_options_print", dtype={"printed": str})
    assert [fenv._r_digits(x) for x in printed["x"]] == printed["printed"].tolist()
    names = ["display.max_columns", "display.max_rows", "display.width", "display.float_format"]
    saved, numpy_saved = {name: pd.get_option(name) for name in names}, np.get_printoptions()
    try:
        with warnings.catch_warnings():
            fenv.environment_options()
        assert repr(pd.DataFrame({"x": [16884.924, 0.000012345]})).split() == ["x", "0", "16885", "1", "0.00001234"]
        assert np.array2string(np.array([16884.924, 1.5])) == "[16885 1.5]"
    finally:
        for name, value in saved.items():
            pd.set_option(name, value)
        np.set_printoptions(**numpy_saved)


def test_get_fwd_and_get_script_directory(tmp_path, monkeypatch):
    """rwf's "--file=" argument first (with Rscript's "~+~" for spaces in get_fwd), and "/" separators with a
    trailing "/" for the directory, as R's dirname() and getwd()."""
    script = tmp_path / "my script.py"
    script.write_text("")
    monkeypatch.setattr(sys, "argv", ["python", "--file=" + str(script).replace(" ", "~+~")])
    assert fenv.get_fwd() == os.path.realpath(script)
    assert fenv.getfwp is fenv.get_fwd
    monkeypatch.setattr(sys, "argv", ["python", "--file=" + str(script)])
    assert functions.get_script_directory() == os.path.realpath(tmp_path).replace(os.sep, "/") + "/"
    monkeypatch.setattr(sys, "argv", ["python"])
    monkeypatch.setattr(sys.modules["__main__"], "__file__", None, raising=False)
    monkeypatch.chdir(tmp_path)
    assert fenv.get_fwd() == ""
    assert functions.get_script_directory() == os.getcwd().replace(os.sep, "/") + "/"


def test_detach_package_reloads_the_module(tmp_path, monkeypatch):
    # counts how many times its code runs
    (tmp_path / "pwf_detach_probe.py").write_text(
        "import builtins\nbuiltins._pwf_probe = getattr(builtins, '_pwf_probe', 0) + 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    import builtins
    try:
        __import__("pwf_detach_probe")
        assert "pwf_detach_probe" in sys.modules
        fenv.detach_package("pwf_detach_probe")
        assert "pwf_detach_probe" not in sys.modules
        __import__("pwf_detach_probe")  # imported again: its code runs again, as after R's detach(unload = TRUE)
        assert builtins._pwf_probe == 2
        fenv.detach_package("pwf_not_imported")  # nothing to do, as R
    finally:
        sys.modules.pop("pwf_detach_probe", None)
        if hasattr(builtins, "_pwf_probe"):
            del builtins._pwf_probe


def test_install_load_and_the_package_stubs(monkeypatch, capsys):
    """install_load returns rwf's named TRUE/FALSE per package; pip is not run here."""
    calls = []

    class Done:
        stdout = "numpy==2.0\npandas==3.0\n"

    monkeypatch.setattr(fenv.subprocess, "run", lambda *args, **kwargs: calls.append(args[0]) or Done())
    assert fenv.install_load("json") == {"json": True}
    assert fenv.install_load(["json", "pwf_no_such_package"]) == {"json": True, "pwf_no_such_package": False}
    assert calls == [[sys.executable, "-m", "pip", "install", "pwf_no_such_package"]]
    with pytest.warns(UserWarning, match="no safe Python equivalent"):
        assert fenv.install_all_packages() is None
    with pytest.warns(UserWarning, match="confirm=True"):
        assert fenv.remove_user_packages() is None
    assert fenv.remove_user_packages(confirm=True) == ["numpy", "pandas"]  # listed, never uninstalled
    assert "not actually uninstalling" in capsys.readouterr().out


def test_write_txt_appends_and_echoes_the_log(tmp_path, capsys):
    """rwf's sink(append = TRUE): every call adds to <file>.log and echoes the whole log."""
    file = str(tmp_path / "log")
    fenv.write_txt("first", file=file)
    fenv.write_txt({"b": 2}, file=file)
    assert capsys.readouterr().out == "first\nfirst\n{'b': 2}\n"
    assert (tmp_path / "log.log").read_text().splitlines() == ["first", "{'b': 2}"]
    fenv.write_txt("console only")
    assert capsys.readouterr().out == "console only\n"
