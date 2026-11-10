# -*- coding: utf-8 -*-
"""
Python port of R rwf::FUNCTIONS_ENVIRONMENT.R.

Deviations from the R originals, by design:
  - install_all_packages()/remove_user_packages() have no sensible or
    safe Python equivalent (PyPI has ~500k+ packages vs CRAN's much
    smaller curated set, and "uninstall everything non-stdlib" would
    wreck the running environment). Both are implemented as guarded
    stubs that require an explicit confirm=True and print a warning
    instead of silently acting.
"""
##########################################################################################
# LOAD SYSTEM
##########################################################################################
import io
import importlib
import os
import re
import subprocess
import sys
import warnings
import numpy as np
import pandas as pd
##########################################################################################
# LOAD ENVIRONMENT
##########################################################################################
def environment_options():
    """
    Set session-wide display/warning options, the Python analogue of R's
    global options() calls: show all DataFrame columns/rows, avoid
    scientific notation, cap float display precision, and reset the
    warnings filter to its default behavior.

    Returns:
    None.

    Examples:
    >>> environment_options()
    """
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', lambda x: f'{x:.4g}')
    np.set_printoptions(suppress=True, precision=4)
    warnings.filterwarnings('default')
##########################################################################################
# LOAD AND INSTALL MULTIPLE LIBRARIES
##########################################################################################
def install_load(package):
    """
    Install (via pip, if not already importable) and import each named
    package.

    Parameters:
    package (str or list of str): Package/module name(s) to install (if
        needed) and import.

    Returns:
    dict: {package_name: bool}, True if the package was successfully
    imported, False if importing (even after attempting installation)
    failed.

    Examples:
    >>> install_load("json")
    >>> install_load(["json", "numpy"])
    """
    if isinstance(package, str):
        package = [package]

    result = {}
    for pkg in package:
        try:
            importlib.import_module(pkg)
            result[pkg] = True
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", pkg], check=False)
            try:
                importlib.import_module(pkg)
                result[pkg] = True
            except ImportError:
                result[pkg] = False
    return result
##########################################################################################
# INSTALL ALL PACKAGES
##########################################################################################
def install_all_packages(confirm=False):
    """
    Guarded stub for R's install_all_packages(), which installs every
    package on CRAN not yet installed. There's no sensible Python
    equivalent — PyPI has ~500k+ packages, so "install everything
    available" isn't a meaningful operation for pip. This function
    exists only to document that fact and never installs anything.

    Parameters:
    confirm (bool, optional): Ignored — always refuses. Present only so
        a call site matching the R signature's intent doesn't silently
        no-op without explanation. Defaults to False.

    Returns:
    None.

    Examples:
    >>> install_all_packages()
    """
    warnings.warn(
        "install_all_packages() has no safe Python equivalent (PyPI has "
        "~500k+ packages; 'install everything available' is not a "
        "meaningful pip operation) and will not install anything.",
        stacklevel=2,
    )
    return None
##########################################################################################
# REMOVE USER INSTALLED PACKAGES
##########################################################################################
def remove_user_packages(confirm=False):
    """
    Guarded stub for R's remove_user_packages(), which uninstalls every
    non-base/recommended package. A literal Python port would uninstall
    everything in the current environment except the standard library —
    this function requires an explicit confirm=True and still only
    prints what it *would* remove, rather than actually uninstalling
    anything, since that action is too destructive to perform silently.

    Parameters:
    confirm (bool, optional): Must be True to do anything at all.
        Defaults to False.

    Returns:
    list of str or None: The list of installed third-party package names
    that would be removed, if confirm=True; otherwise None.

    Examples:
    >>> remove_user_packages()
    >>> remove_user_packages(confirm=True)
    """
    if not confirm:
        warnings.warn(
            "remove_user_packages() would uninstall every third-party "
            "package in this environment. Call with confirm=True to see "
            "what would be removed (this still does not uninstall "
            "anything — that step is intentionally not automated).",
            stacklevel=2,
        )
        return None

    result = subprocess.run([sys.executable, "-m", "pip", "list", "--format=freeze"],
                             capture_output=True, text=True, check=False)
    packages = [line.split("==")[0] for line in result.stdout.splitlines() if line]
    print(f"Would remove {len(packages)} packages (not actually uninstalling):")
    for pkg in packages:
        print(f"  {pkg}")
    return packages
##########################################################################################
# UNLOAD LIBRARY
##########################################################################################
def detach_package(package):
    """
    Remove a package (and any of its submodules) from sys.modules, so
    the next `import` re-executes its module code from scratch. Does
    nothing if the package isn't currently imported.

    Parameters:
    package (str): Name of the package/module to unload, e.g. "numpy".

    Returns:
    None.

    Examples:
    >>> import json
    >>> detach_package("json")
    """
    to_remove = [name for name in sys.modules
                 if name == package or name.startswith(package + '.')]
    for name in to_remove:
        del sys.modules[name]
##########################################################################################
# COMBINE FILES
##########################################################################################
def combine_files(input_dir="functions", output_file="all_functions.py", pattern=r"\.py$", recursive=True):
    """
    Concatenate every file in input_dir whose name matches pattern into one
    output file, each preceded by a banner with its path (port of
    rwf::combine_files). The output file itself is skipped if it lies inside
    input_dir. Defaults are adapted to Python sources; rwf's are
    "working_functions", "all_functions.R" and "\\.[Rr]$".

    Parameters:
    input_dir (str, optional): Directory to read. Defaults to "functions".
    output_file (str, optional): File to write. Defaults to "all_functions.py".
    pattern (str, optional): Regular expression matched against file names
        (not paths). Defaults to r"\\.py$".
    recursive (bool, optional): Include subdirectories. Defaults to True.

    Returns:
    list of str: Paths of the files combined, in the order written.

    Examples:
    >>> combine_files(input_dir="functions", output_file="all_functions.py")
    """
    regex = re.compile(pattern)
    files = []
    for root, dirs, names in os.walk(input_dir):
        files.extend(os.path.join(root, name).replace(os.sep, "/") for name in names if regex.search(name))
        if not recursive:
            break
    output_path = os.path.realpath(output_file)
    files = sorted(f for f in files if os.path.realpath(f) != output_path)
    lines = []
    for f in files:
        with open(f, encoding="utf-8", errors="replace") as handle:
            content = handle.read().splitlines()
        lines += ["#" * 90, f"# FILE: {f}", "#" * 90, *content, ""]
    with open(output_file, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("".join(line + "\n" for line in lines))
    print("Combined", len(files), "files into", output_file)
    return files
##########################################################################################
# GET WORKING FILE PATH
##########################################################################################
def get_fwd():
    """
    Absolute path of the currently running script (port of rwf::get_fwd).
    Tries, in order: a "--file=" command-line argument (as Rscript passes),
    then the __main__ module's __file__. Returns "" when the path can't be
    determined, e.g. in an interactive session. rwf's RStudio fallbacks have
    no Python equivalent.

    Returns:
    str: Absolute path of the running script, or "".

    Examples:
    >>> get_fwd()
    """
    for arg in sys.argv:
        if arg.startswith("--file="):
            return os.path.realpath(arg[len("--file="):].replace("~+~", " "))

    main_module = sys.modules.get("__main__")
    if main_module is not None and getattr(main_module, "__file__", None):
        return os.path.realpath(main_module.__file__)

    return ""


getfwp = get_fwd  # earlier pwf name, kept so existing code keeps working
##########################################################################################
# LOG FILE
##########################################################################################
def write_txt(input, file=None):
    """
    Print `input`. When `file` is given, the printed output is also
    appended to "<file>.log", and the *entire accumulated log file* is
    then echoed back to the console — matching R's sink(append=TRUE)
    behavior, where repeated calls with the same file keep growing the
    log and each call re-displays everything written so far, not just
    the newest addition.

    Parameters:
    input: Any printable object.
    file (str, optional): Log filename without extension. When None
        (default), output goes to the console only. Defaults to None.

    Returns:
    None.

    Examples:
    >>> write_txt("hello")
    >>> write_txt("hello", file="example")
    """
    if file is not None:
        buf = io.StringIO()
        print(input, file=buf)
        logpath = f"{file}.log"
        with open(logpath, 'a') as f:
            f.write(buf.getvalue())
        with open(logpath) as f:
            print(f.read(), end="")
    else:
        print(input)
##########################################################################################
# EXAMPLES
##########################################################################################
if __name__ == "__main__":
    print("=" * 80, "\nenvironment_options\n", "=" * 80, sep="")
    environment_options()
    print("pd.get_option('display.max_columns'):", pd.get_option('display.max_columns'))

    print("\n" + "=" * 80, "\ninstall_load\n", "=" * 80, sep="")
    print(install_load(["json", "numpy"]))

    print("\n" + "=" * 80, "\ninstall_all_packages (guarded, does nothing)\n", "=" * 80, sep="")
    install_all_packages()

    print("\n" + "=" * 80, "\nremove_user_packages (guarded, does nothing without confirm)\n", "=" * 80, sep="")
    print(remove_user_packages())

    print("\n" + "=" * 80, "\ndetach_package\n", "=" * 80, sep="")
    import json  # noqa: F401
    print("json" in sys.modules)
    detach_package("json")
    print("json" in sys.modules)

    print("\n" + "=" * 80, "\nget_fwd\n", "=" * 80, sep="")
    print(get_fwd())

    print("\n" + "=" * 80, "\nwrite_txt\n", "=" * 80, sep="")
    write_txt("hello console only")
    write_txt({"a": 1, "b": 2}, file="write_txt_example")
    write_txt({"c": 3}, file="write_txt_example")
