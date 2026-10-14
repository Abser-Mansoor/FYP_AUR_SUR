"""Python ports of the official MATLAB evaluation scripts.

- tinyface_identification  <- tinyface/Face_Identification_Evaluation/test_face_identification.m + compute_AP.m
- survface_openset         <- QMUL-SurvFace/Face_Identification_Evaluation/test_open_set_face_identification.m
- survface_verification    <- QMUL-SurvFace/Face_Verification_Evaluation/test_face_verification.m
- lfw_verification         <- standard 10-fold LFW protocol (pairs.txt order), cosine similarity

Each protocol is split into (a) a per-sample score pass on embeddings and
(b) a `*_from_scores` function computing the official metrics from those
per-sample scores. (b) is cheap, so bootstrap CIs resample the per-sample
scores and re-run (b). Point estimates use exactly the official arithmetic.

Inputs are float32 numpy embeddings; they are L2-normalized here, as the
official scripts assume normalized features with Euclidean distance.
Returned dicts hold metrics; per-sample arrays are under the "_per_sample" key.
"""
import numpy as np
import torch

_trapz = getattr(np, "trapezoid", None) or np.trapz


def l2n(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _pick(fars, tars, target):
    # MATLAB: sort(abs(FARs-target)) and take the first (stable) index
    return float(tars[np.argsort(np.abs(fars - target), kind="stable")[0]])


def _count_below(sorted_x, th):
    """#(x < th) for each threshold (MATLAB: sum(x < threshold))."""
    return np.searchsorted(sorted_x, th, side="left")


# ------------------------------------------------------------------ TinyFace
def tinyface_from_scores(first_rank, ap):
    out = {"mAP": float(np.mean(ap))}
    for k in (1, 5, 10, 20):
        out[f"Rank-{k}"] = float(np.mean(first_rank < k))
    return out


def tinyface_identification(probe, probe_ids, gallery, gallery_ids, distractor, device="cuda", chunk=256):
    """Closed-set identification with distractors. Returns mAP and CMC@1/5/10/20.

    Per sample (probe): first_rank (0-based rank of the first true match), ap,
    genuine (best cosine to a true gallery image), impostor (best cosine to any other gallery image).
    """
    G = torch.tensor(l2n(np.vstack([gallery, distractor])), device=device)
    gid = torch.tensor(np.concatenate([gallery_ids, -100 * np.ones(len(distractor), np.int64)]), device=device)
    P = torch.tensor(l2n(probe), device=device)
    pid = torch.tensor(np.asarray(probe_ids, np.int64), device=device)
    aps, first, gen, imp = [], [], [], []
    for s in range(0, len(P), chunk):
        S = P[s:s + chunk] @ G.T  # higher = closer (monotone with Euclidean on unit vectors)
        for i in range(S.shape[0]):
            is_good = gid == pid[s + i]
            sg = S[i, is_good]
            # 0-based rank position of each good image in the descending-similarity ordering
            r = (S[i][None, :] > sg[:, None]).sum(1).sort().values.cpu().numpy()
            k = len(r)
            hits = np.arange(1, k + 1)
            prec = hits / (r + 1)
            prev = np.where(r == 0, 1.0, (hits - 1) / np.maximum(r, 1))
            aps.append(float(np.sum((prev + prec) / 2) / k))
            first.append(int(r[0]))
            gen.append(float(sg.max()))
            imp.append(float(S[i].masked_fill(is_good, -2.0).max()))
    first, aps = np.array(first), np.array(aps)
    out = tinyface_from_scores(first, aps)
    out.update(n_probe=len(P), n_gallery=len(G))
    out["_per_sample"] = dict(first_rank=first, ap=aps, genuine=np.array(gen), impostor=np.array(imp))
    return out


# ------------------------------------------------------------ SurvFace 1:N
def survface_openset_from_scores(s1, gt):
    """s1: top-1 distance of each unmated query; gt: distance of the first true
    match within the top-20 for each mated query (1e5 if not in the top-20)."""
    s1s, gts = np.sort(s1), np.sort(gt)
    step = (s1s[-1] - s1s[0]) / 1000
    th = np.arange(s1s[0], s1s[-1] + step / 2, step)
    fpir = _count_below(s1s, th) / len(s1s)
    tpir = _count_below(gts, th) / len(gts)
    o = np.argsort(fpir, kind="stable")
    fpir, tpir = fpir[o], tpir[o]
    return {
        "TPIR20@FPIR=0.3": _pick(fpir, tpir, 0.3),
        "TPIR20@FPIR=0.2": _pick(fpir, tpir, 0.2),
        "TPIR20@FPIR=0.1": _pick(fpir, tpir, 0.1),
        "AUC": float(_trapz(tpir, fpir)),
    }


def survface_openset(gallery, gallery_ids, mated, mated_ids, unmated, device="cuda", L=20):
    """Open-set identification: TPIR20@FPIR=0.3/0.2/0.1 and AUC.

    Per sample: s1 (unmated), gt (mated), plus cosine genuine (mated query vs its
    normalized template) and impostor (unmated query, best cosine to any template).
    """
    gallery = l2n(gallery)
    gallery_ids = np.asarray(gallery_ids)
    tids = np.unique(gallery_ids)
    T = np.stack([gallery[gallery_ids == i].mean(0) for i in tids])  # construct_template.m: plain mean
    M, U = l2n(mated), l2n(unmated)
    Tt = torch.tensor(T, device=device)
    L = min(L, len(tids))
    s_l, ix_l = [], []
    for X in (M, U):
        for s in range(0, len(X), 8192):
            D = torch.cdist(torch.tensor(X[s:s + 8192], device=device), Tt)
            d, i = D.topk(L, largest=False)
            s_l.append(d.cpu().numpy()); ix_l.append(i.cpu().numpy())
    s, ix = np.concatenate(s_l), np.concatenate(ix_l)
    qid = np.concatenate([np.asarray(mated_ids), -100 * np.ones(len(U), np.int64)])
    neg = qid == -100
    s1 = s[neg, 0]
    hit = tids[ix[~neg]] == qid[~neg, None]
    pos_s = s[~neg]
    gt = np.where(hit.any(1), pos_s[np.arange(len(pos_s)), hit.argmax(1)], 1e5)
    out = survface_openset_from_scores(s1, gt)
    out.update(n_gallery_templates=len(tids), n_mated_queries=int((~neg).sum()), n_unmated_queries=int(neg.sum()))
    Tn = l2n(T)
    tpos = {t: i for i, t in enumerate(tids)}
    genuine = (M * Tn[[tpos[i] for i in mated_ids]]).sum(1)
    out["_per_sample"] = dict(s1=s1, gt=gt, genuine=genuine, impostor=_best_cos(U, Tn, device))
    return out


def _best_cos(X, T, device):
    Tt = torch.tensor(T, device=device)
    return np.concatenate([(torch.tensor(X[s:s + 8192], device=device) @ Tt.T).max(1).values.cpu().numpy()
                           for s in range(0, len(X), 8192)])


# ------------------------------------------------------------ SurvFace 1:1
_VER_TH = np.arange(0.2, 2.0 + 1e-9, 0.001)


def survface_verification_from_scores(pos, neg, accuracy=True):
    """pos/neg: Euclidean distances of normalized features (smaller = more similar)."""
    ps, ns = np.sort(pos), np.sort(neg)
    fars = _count_below(ns, _VER_TH) / len(ns)
    tars = _count_below(ps, _VER_TH) / len(ps)
    o = np.argsort(fars, kind="stable")
    fars, tars = fars[o], tars[o]
    out = {f"TAR@FAR={f}": _pick(fars, tars, f) for f in (0.3, 0.1, 0.01, 0.001)}
    out["AUC"] = float(_trapz(tars, fars))
    if accuracy:
        n = len(pos) // 10
        accs = []
        for i in range(10):
            te = np.zeros(len(pos), bool)
            te[i * n:(i + 1) * n] = True
            ptr, ntr = np.sort(pos[~te]), np.sort(neg[~te])
            # sum(train_p < t) + sum(train_n > t)
            acc = (_count_below(ptr, _VER_TH) + len(ntr) - np.searchsorted(ntr, _VER_TH, side="right")) / (len(ptr) + len(ntr))
            t = _VER_TH[np.argmax(acc)]  # first max, as MATLAB's strict '>' keeps the first
            accs.append(((pos[te] < t).sum() + (neg[te] > t).sum()) / (2 * te.sum()))
        out["mean_accuracy"] = float(np.mean(accs))
    return out


def survface_verification(pos_a, pos_b, neg_a, neg_b):
    """Euclidean distance on normalized features; thresholds 0.2:0.001:2.0 as in the .m file."""
    pos = np.linalg.norm(l2n(pos_a) - l2n(pos_b), axis=1)
    neg = np.linalg.norm(l2n(neg_a) - l2n(neg_b), axis=1)
    out = survface_verification_from_scores(pos, neg)
    out.update(n_pos_pairs=len(pos), n_neg_pairs=len(neg))
    # cosine = 1 - d^2/2 on unit vectors
    out["_per_sample"] = dict(pos_dist=pos, neg_dist=neg, genuine=1 - pos ** 2 / 2, impostor=1 - neg ** 2 / 2)
    return out


# ------------------------------------------------------------------- LFW
def lfw_verification(a, b, same, folds=10):
    """Standard LFW 10-fold accuracy; pairs.txt is ordered in 10 folds of 600."""
    sim = (l2n(a) * l2n(b)).sum(1)
    same = np.asarray(same, bool)
    n = len(sim) // folds
    th = np.arange(-1, 1, 0.001)
    accs = []
    for i in range(folds):
        te = np.zeros(len(sim), bool)
        te[i * n:(i + 1) * n] = True
        acc = ((sim[~te][None] > th[:, None]) == same[~te][None]).mean(1)
        t = th[np.argmax(acc)]
        accs.append(((sim[te] > t) == same[te]).mean())
    return {"accuracy": float(np.mean(accs)), "std": float(np.std(accs))}
