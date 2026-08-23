"""
Analyse statistique APPARIEE AU NIVEAU PUZZLE des sorties eval_limem.py.

Principe (cf. discussion methodo) :
  - Les 8 tirages d'un puzzle sont correles -> le N effectif est ~ le nb de
    PUZZLES, pas le nb de generations. On resume donc chaque puzzle a UN taux
    par condition (moyenne sur ses tirages), puis on travaille sur les
    differences appariees puzzle-par-puzzle (unites iid).
  - Test primaire  : bootstrap apparie par cluster (aucune hypothese de forme,
    gere le clustering en reechantillonnant les puzzles).
  - Contre-verif   : t-test apparie (moyenne des differences ; normalite de la
    moyenne assuree par le CLT a n~300, pas besoin de differences normales).
  - Secondaire     : McNemar sur un binaire derive pass@k (1 obs/puzzle/cond).
  - Wilcoxon       : robustesse seulement (interpretation propre => differences
    symetriques, condition NON garantie ici -> a lire avec prudence).
  - Descriptif     : ICC + design-effect pour materialiser le clustering.
  - Comparaisons multiples : correction Holm sur la famille de p-values.

Modes :
  # RL vs SFT sur une condition (entre deux fichiers, appariement par puzzle_id)
  python Solver/paired_stats_limem.py between \
      --a Solver/results/limem_qwen-sft-v2-train-limem.jsonl \
      --b Solver/results/limem_qwen-sftrl-train-limem.jsonl \
      --condition canonique

  # Gap LiMem interne (canonique vs isomorphe) dans un fichier
  python Solver/paired_stats_limem.py gap \
      --a Solver/results/limem_qwen-sft-v2-train-limem.jsonl

  # Memory-leak vs hasard 1/3 (un seul fichier, condition mutant)
  python Solver/paired_stats_limem.py leak \
      --a Solver/results/limem_qwen-sftrl-train-limem.jsonl
"""
import os
import json
import math
import argparse
import collections

import numpy as np

try:
    from scipy import stats as _sp
except Exception:                      # scipy optionnel : fallback normal
    _sp = None

CHANCE_LEAK = 1.0 / 3.0
BOOT = 10000
BOOT_SEED = 12345

# Metriques binaires (moyenne = taux) ou continues stockees par record
BIN_FIELDS = {"gem": "gem_correct", "strict": "strict_correct", "leak": "memory_leak"}
CONT_FIELDS = {"bit": "bit_accuracy"}


# ---------------------------------------------------------------- I/O
def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def per_puzzle(rows, condition, field):
    """puzzle_id -> liste des valeurs (par tirage) pour (condition, field)."""
    buckets = collections.defaultdict(list)
    for r in rows:
        if r.get("condition") != condition:
            continue
        v = r.get(field)
        if v is None:
            continue
        buckets[str(r["puzzle_id"])].append(float(bool(v)) if not isinstance(v, (int, float)) else float(v))
    return buckets


# ---------------------------------------------------------------- stats
def paired_bootstrap(diffs, n_boot=BOOT, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    n = len(diffs)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    # p bilateral : proportion de tirages boot du "mauvais" cote de 0, x2
    p = 2.0 * min((means <= 0).mean(), (means >= 0).mean())
    return float(np.clip(p, 0, 1)), (float(lo), float(hi))


def paired_t(diffs):
    n = len(diffs)
    m = float(diffs.mean())
    sd = float(diffs.std(ddof=1))
    if sd == 0:
        return m, 0.0, (1.0 if m == 0 else 0.0), 0.0
    se = sd / math.sqrt(n)
    t = m / se
    if _sp is not None:
        p = float(2 * _sp.t.sf(abs(t), df=n - 1))
    else:
        p = float(2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2)))))
    d = m / sd                          # Cohen's d apparie
    return m, p, d, se


def wilcoxon(diffs):
    nz = diffs[diffs != 0]
    if _sp is None or len(nz) < 10:
        return None
    try:
        return float(_sp.wilcoxon(nz).pvalue)
    except Exception:
        return None


def mcnemar_passk(a_bucket, b_bucket, ids, k_thresh=1):
    """McNemar sur binaire derive pass@k : 1 si >=k_thresh succes sur les tirages."""
    b = c = 0
    for pid in ids:
        pa = int(sum(a_bucket[pid]) >= k_thresh)
        pb = int(sum(b_bucket[pid]) >= k_thresh)
        if pa == 1 and pb == 0:
            b += 1
        elif pa == 0 and pb == 1:
            c += 1
    ndis = b + c
    if ndis == 0:
        return b, c, 1.0
    if _sp is not None:                 # binomial exact
        p = float(_sp.binomtest(b, ndis, 0.5).pvalue)
    else:
        # approx normale a correction de continuite
        chi = (abs(b - c) - 1) ** 2 / ndis
        p = float(math.erfc(math.sqrt(chi / 2)))
    return b, c, p


def icc_designeffect(buckets, ids):
    """ICC one-way + design-effect, DESCRIPTIF (pas un test)."""
    groups = [np.array(buckets[pid]) for pid in ids if len(buckets[pid]) > 0]
    k = len(groups)
    ni = np.array([len(g) for g in groups])
    if k < 2 or ni.sum() == 0:
        return None
    grand = np.concatenate(groups)
    gm = grand.mean()
    ssb = sum(len(g) * (g.mean() - gm) ** 2 for g in groups)
    ssw = sum(((g - g.mean()) ** 2).sum() for g in groups)
    N = ni.sum()
    msb = ssb / (k - 1)
    msw = ssw / (N - k) if N > k else 0.0
    mbar = (N - (ni ** 2).sum() / N) / (k - 1)   # taille de groupe ajustee
    denom = msb + (mbar - 1) * msw
    icc = (msb - msw) / denom if denom > 0 else 0.0
    deff = 1 + (ni.mean() - 1) * max(icc, 0.0)
    return icc, deff, ni.mean(), k


def holm(pairs):
    """pairs = [(nom, p)] -> [(nom, p, p_ajuste, rejet@0.05)]. Holm-Bonferroni."""
    order = sorted(pairs, key=lambda x: x[1])
    m = len(order)
    out = []
    running = 0.0
    for i, (name, p) in enumerate(order):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)     # monotonicite
        out.append((name, p, running, running < 0.05))
    return out


# ---------------------------------------------------------------- rendu
def show_contrast(title, diffs, extra=None):
    d = np.asarray(diffs, float)
    n = len(d)
    m, p_t, cohd, se = paired_t(d)
    p_b, (lo, hi) = paired_bootstrap(d)
    p_w = wilcoxon(d)
    print(f"\n  {title}")
    print(f"    n(puzzles apparies) = {n}")
    print(f"    diff moyenne        = {m*100:+.2f} pts   (SE {se*100:.2f})")
    print(f"    IC95 bootstrap      = [{lo*100:+.2f}, {hi*100:+.2f}] pts")
    print(f"    p (bootstrap)       = {p_b:.4f}   [primaire]")
    print(f"    p (t apparie)       = {p_t:.4f}   Cohen d={cohd:+.3f}")
    if p_w is not None:
        print(f"    p (Wilcoxon)        = {p_w:.4f}   [robustesse, cf. symetrie]")
    if extra:
        print("    " + extra)
    return p_b


def align_ids(*buckets):
    common = set(buckets[0])
    for b in buckets[1:]:
        common &= set(b)
    return sorted(common)


def mode_between(args):
    ra, rb = load(args.a), load(args.b)
    fam = []
    print("=" * 78)
    print(f"  RL vs SFT — condition '{args.condition}'  (A={os.path.basename(args.a)}  B={os.path.basename(args.b)})")
    print("=" * 78)
    for key, field in list(CONT_FIELDS.items()) + list(BIN_FIELDS.items()):
        if key == "leak":
            continue
        ba = per_puzzle(ra, args.condition, field)
        bb = per_puzzle(rb, args.condition, field)
        ids = align_ids(ba, bb)
        if not ids:
            continue
        diffs = np.array([np.mean(bb[i]) - np.mean(ba[i]) for i in ids])  # B - A = RL - SFT
        icc = icc_designeffect(bb, ids)
        extra = None
        if icc:
            extra = f"ICC(B)={icc[0]:.3f}  design-effect={icc[1]:.2f}  (m={icc[2]:.1f}/puzzle)"
        mcn = ""
        if key in BIN_FIELDS:
            b, c, pmc = mcnemar_passk(ba, bb, ids, k_thresh=1)
            mcn = f"McNemar pass@1: discordants {b}(SFT) / {c}(RL), p={pmc:.4f}"
        p = show_contrast(f"[{key}]  {field}  (RL - SFT)", diffs,
                          extra=(extra + ("\n    " + mcn if mcn else "")) if extra else mcn)
        fam.append((f"{key}:{args.condition}", p))
    _print_holm(fam)


def mode_gap(args):
    r = load(args.a)
    fam = []
    print("=" * 78)
    print(f"  Gap LiMem (canonique - isomorphe)  —  {os.path.basename(args.a)}")
    print("=" * 78)
    for key, field in list(CONT_FIELDS.items()) + list(BIN_FIELDS.items()):
        if key == "leak":
            continue
        bc = per_puzzle(r, "canonique", field)
        bi = per_puzzle(r, "isomorphe", field)
        ids = align_ids(bc, bi)
        if not ids:
            continue
        diffs = np.array([np.mean(bc[i]) - np.mean(bi[i]) for i in ids])  # can - iso
        p = show_contrast(f"[{key}]  {field}  (canonique - isomorphe)", diffs)
        fam.append((f"gap:{key}", p))
    _print_holm(fam)


def mode_leak(args):
    r = load(args.a)
    print("=" * 78)
    print(f"  Memory-leak vs hasard (1/3)  —  {os.path.basename(args.a)}")
    print("=" * 78)
    bl = per_puzzle(r, "mutant", "memory_leak")
    ids = sorted(bl)
    if not ids:
        print("  aucun record mutant.")
        return
    rates = np.array([np.mean(bl[i]) for i in ids])
    diffs = rates - CHANCE_LEAK
    icc = icc_designeffect(bl, ids)
    extra = f"ICC={icc[0]:.3f}  design-effect={icc[1]:.2f}" if icc else None
    print(f"\n  taux de leak moyen = {rates.mean()*100:.2f} %   (hasard {CHANCE_LEAK*100:.1f} %)")
    show_contrast("[leak]  (taux - 1/3)", diffs, extra=extra)
    print("\n  (test one-sample : H0 = leak au hasard ; <0 => re-derive mieux que le hasard)")


def _print_holm(fam):
    if not fam:
        return
    print("\n" + "-" * 78)
    print("  Correction comparaisons multiples (Holm, alpha=0.05) :")
    for name, p, adj, rej in holm(fam):
        print(f"    {name:<22} p={p:.4f}  ->  p_Holm={adj:.4f}  {'REJET H0' if rej else 'n.s.'}")
    print("-" * 78)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    pb = sub.add_parser("between"); pb.add_argument("--a", required=True); pb.add_argument("--b", required=True); pb.add_argument("--condition", default="canonique")
    pg = sub.add_parser("gap"); pg.add_argument("--a", required=True)
    pl = sub.add_parser("leak"); pl.add_argument("--a", required=True)
    args = ap.parse_args()
    {"between": mode_between, "gap": mode_gap, "leak": mode_leak}[args.mode](args)


if __name__ == "__main__":
    main()
