"""
Runtime access to the region files built by data_prep/build_regions.py (see configs/README.md, "Regions").

- Regions:      reads <regions_root>/: manifest, allowed centres, fixed window sets, per-bp cCRE classes.
- TrainSampler: draws training region centres from the pools on the training chromosomes.

A region is the scored interval stored in the files; the dataset builds its input window
(context_length bp) centred on the region.
"""

import json
import os

import numpy as np
import pandas as pd

# cCRE classes in strata/ccre_classes.bed; per-bp strata masks follow this order
STRATA = ["PLS", "ELS", "CTCF", "other"]
TRAIN_SOURCES = {"ccre", "random"}
DEFAULT_TRAIN_WINDOWS = {"sources": {"ccre": 1.0}, "jitter": 200}


def _read_bed(path, names):
    return pd.read_csv(path, sep="\t", comment="#", header=None, names=names)


class Regions:
    def __init__(self, root):
        manifest_path = os.path.join(root, "manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"No region files at {root} (missing manifest.json); run data_prep/setup_data.py.")
        with open(manifest_path) as f:
            self.manifest = json.load(f)
        self.root = root
        self.region_length = self.manifest["region_length"]
        self.max_window = self.manifest["max_window"]

        allowed = _read_bed(os.path.join(root, "allowed_centres.bed"), ["chrom", "start", "end"])
        self.allowed = {c: (g["start"].to_numpy(np.int64), g["end"].to_numpy(np.int64))
                        for c, g in allowed.groupby("chrom", sort=False)}
        self._strata = None  # loaded on first use (eval only)

    def window_set(self, name, max_regions=None, seed=919):
        """Regions of a fixed window set as a list of (chrom, start, end). max_regions: seeded subset (debugging)."""
        path = os.path.join(self.root, "windows", f"{name}.bed")
        if not os.path.exists(path):
            available = sorted(f[:-4] for f in os.listdir(os.path.join(self.root, "windows")) if f.endswith(".bed"))
            raise FileNotFoundError(f"Window set '{name}' not found in {self.root}/windows. Available: {available}. "
                                    "New sets go in data_prep/regions_spec.yaml (then re-run data_prep/setup_data.py).")
        df = _read_bed(path, ["chrom", "start", "end", "region_id", "class"])
        if max_regions is not None and 0 < max_regions < len(df):
            df = df.iloc[np.sort(np.random.default_rng(seed).choice(len(df), size=max_regions, replace=False))]
        return [(c, int(s), int(e)) for c, s, e in zip(df["chrom"], df["start"], df["end"])]

    def is_allowed(self, chrom, centre):
        starts, ends = self.allowed.get(chrom, (np.zeros(0, np.int64), np.zeros(0, np.int64)))
        i = np.searchsorted(starts, centre, side="right") - 1
        return bool(i >= 0 and centre < ends[i])

    def strata_masks(self, chrom, start, end):
        """(len(STRATA), end - start) bool: which bases of [start, end) fall in a cCRE of each class."""
        if self._strata is None:
            df = _read_bed(os.path.join(self.root, "strata", "ccre_classes.bed"), ["chrom", "start", "end", "id", "class"])
            df["k"] = df["class"].map({c: i for i, c in enumerate(STRATA)})
            self._max_ccre = int((df["end"] - df["start"]).max())
            self._strata = {c: (g["start"].to_numpy(np.int64), g["end"].to_numpy(np.int64), g["k"].to_numpy())
                            for c, g in df.sort_values(["chrom", "start"]).groupby("chrom", sort=False)}
        masks = np.zeros((len(STRATA), end - start), dtype=bool)
        if chrom not in self._strata:
            return masks
        s, e, k = self._strata[chrom]
        lo, hi = np.searchsorted(s, start - self._max_ccre), np.searchsorted(s, end)
        for a, b, cls in zip(s[lo:hi], e[lo:hi], k[lo:hi]):
            if b > start:
                masks[cls, max(a, start) - start:min(b, end) - start] = True
        return masks


class TrainSampler:
    """
    Draws training region centres on train_chrs. sources: {"ccre": w, "random": w} (weights, normalised).
      ccre:   a cCRE centre from pools/ccre_centres.bed, plus uniform jitter in [-jitter, jitter]
      random: a uniform position in allowed_centres.bed
    Every returned centre satisfies the region rule (jittered centres outside it are redrawn).
    """
    def __init__(self, regions, train_chrs, sources, jitter=0):
        unknown = set(sources) - TRAIN_SOURCES
        if unknown or not sources or any(w < 0 for w in sources.values()) or sum(sources.values()) <= 0:
            raise ValueError(f"train_windows.sources must map {sorted(TRAIN_SOURCES)} to non-negative weights; got {sources}")
        self.regions, self.jitter = regions, int(jitter)
        names = [s for s in sorted(sources) if sources[s] > 0]
        total = sum(sources[s] for s in names)
        self.sources, self.cum_weights = names, np.cumsum([sources[s] / total for s in names])

        if "ccre" in names:
            pool = _read_bed(os.path.join(regions.root, "pools", "ccre_centres.bed"), ["chrom", "start", "end", "id", "class"])
            pool = pool[pool["chrom"].isin(train_chrs)]
            if pool.empty:
                raise ValueError(f"No cCRE centres on train_chrs {train_chrs}")
            self.ccre_chrom, self.ccre_centre = pool["chrom"].to_numpy(), pool["start"].to_numpy(np.int64)
        if "random" in names:
            chroms = [c for c in train_chrs if c in regions.allowed]
            if not chroms:
                raise ValueError(f"No allowed positions on train_chrs {train_chrs}")
            parts = [(c, *regions.allowed[c]) for c in chroms]
            self.rand_chrom = np.concatenate([[c] * len(s) for c, s, _ in parts])
            self.rand_start = np.concatenate([s for _, s, _ in parts])
            self.rand_cum = np.cumsum(np.concatenate([e - s for _, s, e in parts]))

    def sample_centre(self, rng):
        """(chrom, centre); rng is a random.Random, so draws follow the dataset's per-epoch/worker seed."""
        while True:
            k = int(np.searchsorted(self.cum_weights, rng.random(), side="right"))
            source = self.sources[min(k, len(self.sources) - 1)]  # min: guards float round-off in cum_weights
            if source == "ccre":
                i = rng.randrange(len(self.ccre_centre))
                chrom, centre = self.ccre_chrom[i], int(self.ccre_centre[i])
                if self.jitter:
                    centre += rng.randint(-self.jitter, self.jitter)
                    if not self.regions.is_allowed(chrom, centre):
                        continue
                return chrom, centre
            u = rng.randrange(int(self.rand_cum[-1]))
            k = int(np.searchsorted(self.rand_cum, u, side="right"))
            offset = u - (int(self.rand_cum[k - 1]) if k else 0)
            return self.rand_chrom[k], int(self.rand_start[k]) + offset
