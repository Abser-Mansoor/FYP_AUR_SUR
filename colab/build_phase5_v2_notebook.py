"""Generate colab/phase5_detector_colab_v2.ipynb: the Phase 5 detector study re-run with the official SCRFD preprocessing.

    envs/diag/Scripts/python.exe colab/build_phase5_v2_notebook.py [--publish]

Derived from the first notebook (colab/build_phase5_notebook.py) by exact text replacements, each asserted, so the two cannot drift silently.
--publish copies the notebook to G:/My Drive/FYP_SR_SHARED/phase5_v2/ (a new folder; nothing in phase5/ is overwritten).
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_phase5_notebook as N  # noqa: E402

ROOT = N.ROOT


def sub(cell, old, new):
    src = "".join(N.CELLS[cell]["source"])
    assert src.count(old) == 1, (cell, old[:60], src.count(old))
    N.CELLS[cell]["source"] = src.replace(old, new).splitlines(keepends=True)


# 0: intro
sub(0, "# Phase 5 detector study (Colab T4)", "# Phase 5 detector study, re-run v2 (Colab T4)")
sub(0, "Results go to `FYP_SR_SHARED/phase5/results/` on Drive; the lab PC merges them.",
    "**v2:** SCRFD now uses the official WIDER preprocessing (grey padding after normalisation, rounded resize, NMS 0.45). The data zips are the\n"
    "ones in `phase5/` (unchanged); the code comes from `phase5_v2/code_v2.zip`. Results go to `FYP_SR_SHARED/phase5_v2/results/`; nothing in\n"
    "`phase5/` is written. The lab PC merges them.")
# 1: also verify the v2 manifest
sub(1, 'print("bundle verified:", len(man), "files")',
    'B2 = pathlib.Path(SHARED) / "phase5_v2"\nman2 = json.loads((B2 / "MANIFEST.json").read_text())\n'
    'for rel, meta in man2.items():\n    p = B2 / rel\n    assert p.exists() and p.stat().st_size == meta["bytes"], f"{rel}: missing or incomplete; wait for Drive to finish syncing, then rerun"\n'
    '    if sha256(p) != meta["sha256"]:\n        raise RuntimeError(f"{rel}: checksum mismatch")\n'
    'print("bundle verified:", len(man), "+", len(man2), "files")')
# 2: unpack the old data zips without the old code.zip, then the v2 code
sub(2, "for z in man:\n    if z.endswith(\".zip\"):\n        zipfile.ZipFile(B / z).extractall(WORK)",
    "for z in man:\n    if z.endswith(\".zip\") and z != \"code.zip\":   # the first code.zip is superseded by code_v2.zip\n        zipfile.ZipFile(B / z).extractall(WORK)\n"
    "zipfile.ZipFile(B2 / \"code_v2.zip\").extractall(WORK)")
# 6: results to phase5_v2/results, config out folder detstudy_v2
sub(6, 'OUT = f"{SHARED}/phase5/results"', 'OUT = f"{SHARED}/phase5_v2/results"')
src = "".join(N.CELLS[6]["source"])
assert src.count("outputs/video/detstudy") == 5, src.count("outputs/video/detstudy")
N.CELLS[6]["source"] = src.replace("outputs/video/detstudy", "outputs/video/detstudy_v2").splitlines(keepends=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--publish", action="store_true")
    a = ap.parse_args()
    nb = {"cells": N.CELLS, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                                         "kernelspec": {"display_name": "Python 3", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}
    p = ROOT / "colab/phase5_detector_colab_v2.ipynb"
    p.write_text(json.dumps(nb, indent=1))
    print("wrote", p)
    if a.publish:
        d = N.SHARED / "phase5_v2"
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy(p, d / p.name)
        print("published", d / p.name)


if __name__ == "__main__":
    main()
