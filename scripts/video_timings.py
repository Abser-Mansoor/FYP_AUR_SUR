"""Stage timings of the Phase 6 ChokePoint run, aggregated from the per-clip done_*.json files (800 x 600 frames, lab PC, RTX 4000 Ada).

    envs/video/Scripts/python.exe scripts/video_timings.py
Writes outputs/video/pipeline_timings.csv: per stage, the median over clips and the pooled value (total time / total frames or faces).
The track time is both trackers together; detect and embed include reading and decoding the JPEG frames.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
rows = []
for d in sorted((ROOT / "outputs/video/pipeline/chokepoint").iterdir()):
    if d.is_dir() and (d / "done_embed.json").exists():
        a, t, e = (json.load(open(d / f"done_{s}.json")) for s in ("detect", "track", "embed"))
        rows.append(dict(clip=d.name, frames=a["n_frames"], faces=e["n_embedded"], s_detect=a["seconds_detect"], s_track=t["seconds_track"], s_embed=e["seconds_embed"]))
d = pd.DataFrame(rows)
out = []
for stage, unit, n in (("detect", "frame", "frames"), ("track", "frame", "frames"), ("embed", "face", "faces")):
    per = 1000 * d[f"s_{stage}"] / d[n]
    out.append(dict(stage=stage, unit=unit, n_clips=len(d), n_units=int(d[n].sum()), median_ms_per_unit=per.median(), pooled_ms_per_unit=1000 * d[f"s_{stage}"].sum() / d[n].sum(),
                    p10_clip=per.quantile(0.1), p90_clip=per.quantile(0.9)))
pd.DataFrame(out).to_csv(ROOT / "outputs/video/pipeline_timings.csv", index=False, float_format="%.3f")
print(pd.DataFrame(out).round(3).to_string())
