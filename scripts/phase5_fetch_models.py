"""Download the Phase 5 extra model weights (download and SHA-256 only; nothing is run) into models/.

    envs/video/Scripts/python.exe scripts/phase5_fetch_models.py

  CAFace   mk-minchul/CAFace e94ad0206d0a04a098594747138e4da5435f5ec7 (MIT): CAFace_AdaFaceWebFace4M.ckpt + the class-center file,
           and the AdaFace R100 WebFace4M backbone it was trained on (from the AdaFace README). The backbone is NOT our MS1MV2
           checkpoint, so a CAFace result needs its own single-frame baseline with this backbone.
  FaceMoE  huggingface.co/kartiknarayan/FaceMoE (code: Kartik-3004/FaceMoE f0972d487ed4fe20bce701840e4a3ee2da2f4c4e, MIT); input 120 px.
Writes outputs/video/model_downloads.json (path, bytes, sha256, source). Resumable: finished files are kept.
"""
import hashlib
import json
from pathlib import Path

import gdown

ROOT = Path(__file__).resolve().parents[1]
GDRIVE = {
    "models/caface/CAFace_AdaFaceWebFace4M.ckpt": "1a-nrXcX_h8sLQiTD_mGvmKAG58gE9Oi5",
    "models/caface/center_WebFace4MAdaFace_webface4m_subset.pth": "1WmiWjLSsfQU2PTwQAvnrep9u6Jfvd3tR",
    "models/caface/adaface_ir101_webface4m.ckpt": "18jQkqB0avFqWa0Pas52g54xNshUOQJpQ",
}
HF = ("kartiknarayan/FaceMoE", "models/facemoe", ["swin4m_exp_3_k_2_briar_full/model.pt", "swin4m_exp_3_k_2_tinyface_full/model.pt", "README.md"])


def sha256(p, blk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(blk), b""):
            h.update(b)
    return h.hexdigest()


def main():
    rec = {}
    for rel, fid in GDRIVE.items():
        p = ROOT / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            print("gdown", rel, flush=True)
            gdown.download(id=fid, output=str(p), quiet=True)
        rec[rel] = dict(source=f"https://drive.google.com/file/d/{fid}", bytes=p.stat().st_size if p.exists() else None,
                        sha256=sha256(p) if p.exists() else "DOWNLOAD FAILED")
    from huggingface_hub import hf_hub_download
    for f in HF[2]:
        print("huggingface", f, flush=True)
        p = Path(hf_hub_download(HF[0], f, local_dir=str(ROOT / HF[1])))
        rec[str(p.relative_to(ROOT)).replace("\\", "/")] = dict(source=f"https://huggingface.co/{HF[0]}/blob/main/{f}", bytes=p.stat().st_size, sha256=sha256(p))
    out = ROOT / "outputs/video/model_downloads.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
