"""Recognizer loaders. Backbone definitions are vendored unmodified from the
official repos in src/models/vendor (see vendor/SOURCES.md)."""
import torch

from .vendor import insightface_iresnet, adaface_net, magface_iresnet, curricular_model_irse

# Official input conventions, per each repo's inference code.
#   channel: order fed to the network; norm: "pm1" = (x/255-0.5)/0.5, "01" = x/255
OFFICIAL = {
    "arcface": dict(channel="RGB", norm="pm1"),         # insightface arcface_torch/inference.py
    "adaface": dict(channel="BGR", norm="pm1"),         # AdaFace inference.py to_input()
    "magface": dict(channel="BGR", norm="01"),          # MagFace inference/gen_feat.py (cv2 + ToTensor)
    "curricularface": dict(channel="RGB", norm="pm1"),  # CurricularFace / face.evoLVe eval
}


def _strip(sd, prefix):
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


def load_recognizer(name, weights, device="cuda"):
    """Build backbone, load weights with strict=True, return eval-mode model.

    Every model's forward returns a raw (un-normalized) 512-d embedding.
    """
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    if name == "arcface":
        net = insightface_iresnet.iresnet100()
        sd = ck
    elif name == "adaface":
        net = adaface_net.build_model("ir_101")
        sd = _strip(ck["state_dict"], "model.")
    elif name == "magface":
        net = magface_iresnet.iresnet100(num_classes=512)
        sd = _strip(ck["state_dict"], "features.module.")
    elif name == "curricularface":
        net = curricular_model_irse.IR_101([112, 112])
        sd = ck
    else:
        raise ValueError(name)
    net.load_state_dict(sd, strict=True)
    net.eval().to(device)
    assert not net.training

    def embed(x):
        out = net(x)
        return out[0] if isinstance(out, tuple) else out

    embed.net = net
    return embed
