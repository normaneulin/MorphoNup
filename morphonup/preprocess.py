"""Shape and k-mer preprocessing, fitted on the fit rows of each fold only (forward + RC pooled).

MorphoNup: MGW, Roll, EP kept as z-scored tracks; the 7 RC-symmetric tracks
(Opening, ProT, Stagger, Stretch, HelT, Rise, Slide) reduced by dyad-weighted PLS1 to 6
components, re-standardized on fit rows => 9 channels.
* Weighted PLS1 is weighted least squares: weighted centering, X and y both multiplied by sqrt(w),
  with a Gaussian dyad weight (sigma 35 bp) so the nucleosome centre dominates the fit.
* Reduced scores are re-standardized so every channel enters the RBF prototype distances on the same scale.
* RC-antisymmetric tracks (Buckle, Shear, Shift, Tilt) change sign on the reverse strand.
  With strand pooling and a dyad-symmetric weight their pointwise covariance with the label
  is exactly zero, so a PLS block refuses them; they are dropped or reduced by PCA.
`raw=True` keeps all 14 z-scored tracks in 6 biophysical groups (no reduction; modality control).
"""
import numpy as np

NAMES14 = ["Buckle", "MGW", "Opening", "ProT", "Shear", "Stagger", "Stretch",
           "HelT", "Rise", "Roll", "Shift", "Slide", "Tilt", "EP"]
ANTISYMMETRIC = ["Buckle", "Shear", "Shift", "Tilt"]
# biophysical groups of the 14 channels (used only with raw=True)
RAW_GROUPS = [[0, 2, 3], [4, 5, 6], [1], [7, 9, 12], [8, 10, 11], [13]]


def tracks14(dataset, strand):
    """(N, 143, 14): 7 BP geometry, 6 steps averaged onto BP centres, EP."""
    bp, step, ep = (dataset.arrays[k][strand] for k in ("bp", "step", "ep"))
    step_aligned = 0.5 * (step[:, :-1] + step[:, 1:])
    return np.concatenate([bp, step_aligned, ep], axis=-1).astype(np.float32)


def shape_tracks(dataset, spec):
    """tracks14 of both strands. spec["shuffle_seed"] (control arm): every sequence receives the tracks of another,
    randomly chosen sequence (one fixed permutation for both strands), so shape keeps its values but loses its link
    to the sequence and the label."""
    X = [tracks14(dataset, s) for s in (0, 1)]
    if spec.get("shuffle_seed") is not None:
        perm = np.random.default_rng(spec["shuffle_seed"]).permutation(len(X[0]))
        X = [x[perm] for x in X]
    return X


def dyad_weights(length, sigma):
    if sigma is None:
        return np.ones(length)
    t = np.arange(length)
    w = np.exp(-0.5 * ((t - (length - 1) / 2.0) / sigma) ** 2)
    return w / w.mean()


def weighted_pls1(X, y, w, k):
    """PLS1 (NIPALS) on weighted, weighted-centred data. Returns rotations (p, k)."""
    sw = np.sqrt(w)[:, None]
    mu_x = (w[:, None] * X).sum(0) / w.sum()
    mu_y = (w * y).sum() / w.sum()
    Xd = (X - mu_x) * sw
    yd = (y - mu_y) * sw[:, 0]
    W, P = [], []
    for _ in range(k):
        wk = Xd.T @ yd
        wk /= np.linalg.norm(wk)
        t = Xd @ wk
        tt = t @ t
        p = Xd.T @ t / tt
        Xd -= np.outer(t, p)
        yd -= (yd @ t / tt) * t
        W.append(wk); P.append(p)
    W, P = np.array(W).T, np.array(P).T
    return W @ np.linalg.inv(P.T @ W)


class PhysicalReducer:
    """explicit: channel names kept as z-scored tracks.
    blocks: list of dict(channels=[names], method='pls'|'pca'|'none', k=int); 'none' keeps the z-scored tracks as one
    group without reduction (k = number of channels).
    Channels in neither are dropped. raw=True keeps all 14 in RAW_GROUPS."""

    def __init__(self, explicit=(), blocks=(), raw=False, dyad_sigma=35.0, standardize_scores=True):
        self.raw = raw
        self.explicit = list(range(14)) if raw else [NAMES14.index(n) for n in explicit]
        self.blocks = [] if raw else [dict(idx=[NAMES14.index(n) for n in b["channels"]],
                                           method=b["method"], k=int(b["k"])) for b in blocks]
        used = self.explicit + [i for b in self.blocks for i in b["idx"]]
        if len(used) != len(set(used)):
            raise ValueError("A channel is listed twice.")
        for b in self.blocks:
            if (b["method"] not in ("pls", "pca", "none") or not 1 <= b["k"] <= len(b["idx"])
                    or (b["method"] == "none" and b["k"] != len(b["idx"]))):
                raise ValueError(b)
            if b["method"] == "pls" and any(NAMES14[i] in ANTISYMMETRIC for i in b["idx"]):
                raise ValueError("RC-antisymmetric tracks receive zero pointwise PLS loadings.")
        self.dyad_sigma, self.standardize_scores = dyad_sigma, standardize_scores

    @classmethod
    def from_spec(cls, spec):
        """spec: config.SHAPE-style dict(explicit=[...], blocks=[...], dyad_sigma=..., raw=False)."""
        return cls(spec.get("explicit", ()), spec.get("blocks", ()), raw=spec.get("raw", False),
                   dyad_sigma=spec.get("dyad_sigma", 35.0))

    @property
    def channels(self):
        return len(self.explicit) + sum(b["k"] for b in self.blocks)

    def channel_names(self):
        names = [NAMES14[i] for i in self.explicit]
        for j, b in enumerate(self.blocks):
            names += ([NAMES14[i] for i in b["idx"]] if b["method"] == "none"
                      else [f"{b['method'].upper()}{j}_{c}" for c in range(b["k"])])
        return names

    def groups(self):
        """Prototype/adapter groups: RAW_GROUPS for raw 14; else one per explicit channel + one per block."""
        if self.raw:
            return RAW_GROUPS
        groups = [[j] for j in range(len(self.explicit))]
        start = len(self.explicit)
        for b in self.blocks:
            groups.append(list(range(start, start + b["k"])))
            start += b["k"]
        return groups

    def fit(self, x_fwd, y, x_rc):
        pooled = np.concatenate([x_fwd, x_rc]).astype(np.float64)       # (2N, L, 14)
        n, length, _ = pooled.shape
        flat = pooled.reshape(-1, 14)
        self.mean_, self.std_ = flat.mean(0), np.maximum(flat.std(0), 1e-6)
        z = (flat - self.mean_) / self.std_
        w = np.tile(dyad_weights(length, self.dyad_sigma), n)
        yy = np.repeat(np.concatenate([y, y]).astype(np.float64), length)
        for b in self.blocks:
            zb = z[:, b["idx"]]
            if b["method"] == "pls":
                b["rotation"] = weighted_pls1(zb, yy, w, b["k"])
            elif b["method"] == "none":
                b["rotation"] = np.eye(len(b["idx"]))
            else:
                vals, vecs = np.linalg.eigh(np.cov(zb, rowvar=False))
                order = np.argsort(vals)[::-1]
                b["rotation"] = vecs[:, order[:b["k"]]]
                b["explained"] = vals[order[:b["k"]]] / vals.sum()
            scores = zb @ b["rotation"]
            b["score_mean"] = scores.mean(0)
            b["score_std"] = scores.std(0) if self.standardize_scores else np.ones(b["k"])
        return self

    def transform(self, x):
        n, length, _ = x.shape
        z = (x.reshape(-1, 14).astype(np.float64) - self.mean_) / self.std_
        out = [z[:, self.explicit]]
        for b in self.blocks:
            out.append((z[:, b["idx"]] @ b["rotation"] - b["score_mean"]) / b["score_std"])
        return np.concatenate(out, axis=1).reshape(n, length, self.channels).astype(np.float32)

    def state(self):
        out = dict(mean=self.mean_, std=self.std_, explicit=np.array(self.explicit),
                   channel_names=np.array(self.channel_names()))
        for j, b in enumerate(self.blocks):
            out |= {f"block{j}_{k}": np.asarray(v) for k, v in b.items()}
        return out


def kmer_views(ds, fit):
    """1-4-mer frequencies (340) of both strands, z-scored with fit-row statistics (forward + RC pooled)."""
    raw = [ds.arrays["kmers"][s] for s in (0, 1)]
    pool = np.concatenate([raw[0][fit], raw[1][fit]])
    mu, sd = pool.mean(0), np.maximum(pool.std(0), 1e-6)
    return [((raw[s] - mu) / sd).astype(np.float32) for s in (0, 1)]
