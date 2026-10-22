# Vendored third-party code (src/video/vendor)

Only the files listed here are copied. Each folder has the upstream LICENSE (MIT). Upstream SHA-256 is of the file as downloaded, before the patches below.

| Folder | Upstream | Commit | Licence |
|---|---|---|---|
| bytetrack | github.com/FoundationVision/ByteTrack (`yolox/tracker/`; the old ifzhang/ByteTrack URL redirects here) | d1bf0191adff59bc8fcfeaa0b33d3d1642552a99 | MIT |
| botsort | github.com/NirAharon/BoT-SORT (`tracker/`) | 251985436d6712aaf682aaaf5f71edb4987224bd | MIT |

## Patches (the only changes to upstream files)
- Imports made package-relative (`from yolox.tracker import matching` and `from tracker import ...` become `from . import ...`).
- `np.float` replaced by `float` (removed in NumPy 2).
- `cython_bbox` replaced by `_bbox.py`, a NumPy version with the same +1 pixel convention, so no compiled extension is needed.
- ByteTrack: unused `import torch` lines removed.
- BoT-SORT: top-level `matplotlib` import moved inside the debug branch of `gmc.py`; `fast_reid` import moved inside the `with_reid` branch (re-ID is not used and `fast_reid` is not vendored).

## Upstream SHA-256 (before patches)
```
c938afcf7c516bba4a77155c2499fcae6f926df7a2faa6f73a4be1b073f9bb2d  bytetrack/basetrack.py
7cecdcdd7998103969a4ba1772f4c9fb5560fd5eef05ca03e0d2df28346ca50b  bytetrack/byte_tracker.py
9842d0dd9957c97e312d42e5789441add55082066d87cc051ac768143a46eb39  bytetrack/kalman_filter.py
2f54bf0c183f577087391901562ae2bcd3c01633a3e077bd5d4168fafbd899a6  bytetrack/matching.py
a0f717a5059ebfa626a63cea5b040a649acd7720072f503cabfa24d749f082f6  botsort/basetrack.py
ba6dd32ad4b96ee49d6f70654159269bd45ce15fd0fe5007f3a1b81bc257edf0  botsort/bot_sort.py
508c3a4ea0e9a72478ce2d1d29aa1eb8892b3804c76b524a16a34dca23dba9ba  botsort/gmc.py
c1a4c0acc71b9a91c0cdbe7b689f88712d68e7e8be4d7e54fb437f33fdbc9e5a  botsort/kalman_filter.py
020b2215c20467b31f7d8a977a2f557b8d1838f8baa5a5be54c0b8d14bbee4e1  botsort/matching.py
```

## retinaface
| Folder | Upstream | Commit | Licence |
|---|---|---|---|
| retinaface | github.com/biubug6/Pytorch_Retinaface (`models/`, `layers/functions/prior_box.py`, `utils/box_utils.py`, `utils/nms/py_cpu_nms.py`, `data/config.py`) | b984b4b775b2c4dced95c1eadd195a5c7d32a60b | MIT |

Patch: in `models/retinaface.py`, `from models.net import ...` became `from .net import ...`. Nothing else changed. Weights are not vendored (see `MODEL_INVENTORY.md`).

Upstream SHA-256 (before the patch):
```
dbb885c0d7bd02865c38f6c9f06a03b0c586201a824f797c57ce87ca466a896f *models/net.py
8185ffb26bdf0e10ec83cb286c0834e7a48d4832cb31d7bd7da4810309de6321 *layers/functions/prior_box.py
734588d011237ad33a937f2278a7927a02e299edfa19ac21d38626e75888d260 *utils/box_utils.py
d9f6e53641524e2a77d44af4c880dca4f82006967a5b46063375639a47535336 *utils/nms/py_cpu_nms.py
70a8c14459ee47637785b1537da839e03e30d6500dec89e42fb646447d1349ba *data/config.py
(models/retinaface.py upstream sha256 is recorded below)
```
367da7b58830d340952b5676504b25d3af3b05992793fc29e100d6ebe17a24c9  models/retinaface.py (upstream)
