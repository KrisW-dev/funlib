#!/usr/bin/env python3
"""
Build the release artifacts from the dev tree, and verify them.

Steps (all automatic):
  1. Run test_smoke.py against the dev tree (analysis/ + graphics/ packages).
  2. Concatenate the dev modules into dist/funlib-v<version>/analysis.py
     and dist/funlib-v<version>/graphics.py.
  3. Write dist/funlib-v<version>/README.txt.
  4. Copy the merged files + test_smoke.py to a temp dir and run the
     smoke test there (so the merged files, not the packages, are imported).
  5. Zip dist/funlib-v<version>/ into dist/funlib-v<version>.zip.

Run from the project root:  python build_release.py
Exits nonzero if either smoke test fails.
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

VERSION = "0.3"                     # bump per release; also update CHANGELOG.md
ROOT = os.path.dirname(os.path.abspath(__file__))
DIST = os.path.join(ROOT, "dist")
RELEASE_NAME = f"funlib-v{VERSION}"
RELEASE_DIR = os.path.join(DIST, RELEASE_NAME)

ANALYSIS_MODULES = [
    "analysis/function.py",
]
GRAPHICS_MODULES = [
    "graphics/canvas.py",
    "graphics/func2d.py",
    "graphics/func3d.py",
    "graphics/objects2d.py",
    "graphics/objects3d.py",
    "graphics/__init__.py",         # contributes the registry update + __all__
]


# ---- text helpers ----

def read(relpath):
    with open(os.path.join(ROOT, relpath), encoding="utf-8") as f:
        return f.read()


def strip_internal_imports(text):
    """Drop `from .canvas import ...` / `from .func2d import ...` lines."""
    return re.sub(r"^\s*from \.\w+ import .*$\n?", "", text, flags=re.MULTILINE)


def fix_analysis_imports(text):
    """`from analysis.function import X` -> `from analysis import X`."""
    return re.sub(r"from analysis\.\w+ import", "from analysis import", text)


def build_module(modules, header):
    parts = [header]
    for relpath in modules:
        parts.append(
            "# ============================================================\n"
            f"# {relpath}\n"
            "# ============================================================\n"
        )
        text = read(relpath)
        text = strip_internal_imports(text)
        text = fix_analysis_imports(text)
        parts.append(text)
    return "\n\n".join(parts)


# ---- smoke test runner ----

def run_smoke(cwd, label):
    """Run test_smoke.py in cwd; return True on success."""
    print(f"--- smoke test ({label}) ---")
    result = subprocess.run(
        [sys.executable, "test_smoke.py"],
        cwd=cwd,
        capture_output=False,      # let the per-check output stream through
    )
    if result.returncode != 0:
        print(f"--- smoke test ({label}) FAILED ---")
        return False
    print(f"--- smoke test ({label}) passed ---")
    return True


# ---- build ----

def main():
    # 1. dev-tree smoke test
    if not run_smoke(ROOT, "dev tree"):
        sys.exit(1)

    # 2. build merged files
    os.makedirs(DIST, exist_ok=True)
    if os.path.isdir(RELEASE_DIR):
        shutil.rmtree(RELEASE_DIR)
    os.makedirs(RELEASE_DIR)

    analysis_py = build_module(
        ANALYSIS_MODULES,
        '"""analysis.py — multivariate functions over sympy (generated release file)."""\n',
    )
    graphics_py = build_module(
        GRAPHICS_MODULES,
        '"""graphics.py — canvases, plot classes, geometric objects (generated release file)."""\n',
    )

    with open(os.path.join(RELEASE_DIR, "analysis.py"), "w", encoding="utf-8") as f:
        f.write(analysis_py)
    with open(os.path.join(RELEASE_DIR, "graphics.py"), "w", encoding="utf-8") as f:
        f.write(graphics_py)

    # 3. README
    readme = (
        f"funlib {VERSION}\n"
        f"{'=' * (len('funlib ') + len(VERSION))}\n\n"
        f"Two files: analysis.py, graphics.py. Put them next to your notebook.\n\n"
        f"In a cell:\n\n"
        f"    from analysis import Function\n"
        f"    import graphics\n\n"
        f"Requires: sympy, numpy, matplotlib, scipy.\n"
        f"Optional: ipympl (interactive 3D; run `%matplotlib widget`).\n"
    )
    with open(os.path.join(RELEASE_DIR, "README.txt"), "w", encoding="utf-8") as f:
        f.write(readme)

    # 4. merged-files smoke test in a temp dir
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(os.path.join(RELEASE_DIR, "analysis.py"), tmp)
        shutil.copy(os.path.join(RELEASE_DIR, "graphics.py"), tmp)
        shutil.copy(os.path.join(ROOT, "test_smoke.py"), tmp)
        if not run_smoke(tmp, "merged files"):
            print("Release build aborted: merged-files smoke test failed.")
            sys.exit(1)

    # 5. zip
    zipname = os.path.join(DIST, f"{RELEASE_NAME}.zip")
    with zipfile.ZipFile(zipname, "w", zipfile.ZIP_DEFLATED) as z:
        for name in ("analysis.py", "graphics.py", "README.txt"):
            z.write(os.path.join(RELEASE_DIR, name), arcname=name)

    print()
    print(f"built: {os.path.join(RELEASE_DIR, 'analysis.py')}")
    print(f"built: {os.path.join(RELEASE_DIR, 'graphics.py')}")
    print(f"built: {zipname}")


if __name__ == "__main__":
    main()