#!/usr/bin/env python3
"""
Per-block residual magnitude bisect for QIE Ascend native engine vs CLI ground truth.

Reads CLI block dumps from /tmp/qie_5513f_cli_blocks/blockNN/qie_cli_blkNN_<tag>.f32.bin
and engine block dumps from /tmp/qie_5513f_eng_blocks/blockNN/<tag>.f32

Tags compared:
  13_img_resid1   (post-attention residual, before LN2/FFN)
  24_img_resid2   (post-FFN residual, end-of-block — the per-block residual stream)
  13_txt_resid1
  24_txt_resid2

Reports per-block: cos similarity, ratio of absmax (eng/cli), absmax for each side,
std for each side. Identifies the divergence signature.
"""
import numpy as np
import os
import sys

ENG_DIR = "/tmp/qie_5513f_eng_blocks"
CLI_DIR = "/tmp/qie_5513f_cli_blocks"

BLOCKS = [0,1,2,3,4,5,6,7,8,9,16,24,30,36,45,50,51,52,53,54,55,56,57,58,59]
TAGS_IMG = ["13_img_resid1", "24_img_resid2"]
TAGS_TXT = ["13_txt_resid1", "24_txt_resid2"]

def load_eng(blk, tag):
    p = f"{ENG_DIR}/block{blk:02d}/{tag}.f32"
    if not os.path.exists(p): return None
    return np.fromfile(p, dtype=np.float32)

def load_cli(blk, tag):
    p = f"{CLI_DIR}/block{blk:02d}/qie_cli_blk{blk:02d}_{tag}.f32.bin"
    if not os.path.exists(p): return None
    return np.fromfile(p, dtype=np.float32)

def stats_one(a):
    finite = a[np.isfinite(a)]
    return dict(
        n=len(a),
        n_finite=len(finite),
        nan=int(np.isnan(a).sum()),
        inf=int(np.isinf(a).sum()),
        absmax=float(np.max(np.abs(finite))) if len(finite) else float('nan'),
        std=float(np.std(finite)) if len(finite) else float('nan'),
    )

def cos_sim(a, b):
    fa = np.isfinite(a) & np.isfinite(b)
    if fa.sum() == 0: return float('nan')
    a = a[fa].astype(np.float64)
    b = b[fa].astype(np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0: return float('nan')
    return float(np.dot(a, b) / (na * nb))

def main():
    print(f"{'blk':>3} | {'tag':<14} | {'eng_amax':>10} {'eng_std':>10} | {'cli_amax':>10} {'cli_std':>10} | {'cos':>6} | {'r_amax':>8} | {'eng_n/c':>10}")
    print("-" * 110)
    rows = []
    for blk in BLOCKS:
        for tag in TAGS_IMG + TAGS_TXT:
            eng = load_eng(blk, tag)
            cli = load_cli(blk, tag)
            if eng is None and cli is None: continue
            if eng is None:
                print(f"{blk:>3} | {tag:<14} | {'MISSING':>21} | {stats_one(cli)['absmax']:>10.3g} {stats_one(cli)['std']:>10.3g} |        |          |")
                continue
            if cli is None:
                print(f"{blk:>3} | {tag:<14} | {stats_one(eng)['absmax']:>10.3g} {stats_one(eng)['std']:>10.3g} | {'MISSING':>21} |        |          |")
                continue
            if eng.shape != cli.shape:
                print(f"{blk:>3} | {tag:<14} | shape mismatch eng={eng.shape} cli={cli.shape}")
                continue
            se = stats_one(eng)
            sc = stats_one(cli)
            cos = cos_sim(eng, cli)
            ratio_amax = se['absmax']/sc['absmax'] if sc['absmax'] else float('nan')
            nan_summary = f"{se['nan']}/{sc['nan']}"
            print(f"{blk:>3} | {tag:<14} | {se['absmax']:>10.3g} {se['std']:>10.3g} | {sc['absmax']:>10.3g} {sc['std']:>10.3g} | {cos:>6.3f} | {ratio_amax:>8.3g} | {nan_summary:>10}")
            rows.append((blk, tag, cos, ratio_amax, se['absmax'], sc['absmax']))

    print("\n=== Divergence signature analysis ===")
    # focus on 24_img_resid2 (end-of-block residual stream) — the canonical magnitude metric
    img_r2 = [(blk, cos, r) for (blk, tag, cos, r, _, _) in rows if tag == "24_img_resid2"]
    if img_r2:
        print("\nFor tag 24_img_resid2 (end-of-block img residual):")
        first_cos_under_95 = None
        first_ratio_over_2 = None
        first_ratio_under_05 = None
        for blk, cos, r in img_r2:
            note = []
            if first_cos_under_95 is None and cos < 0.95:
                first_cos_under_95 = blk
                note.append("first cos<0.95")
            if first_ratio_over_2 is None and r >= 2.0:
                first_ratio_over_2 = blk
                note.append("first r>=2")
            if first_ratio_under_05 is None and r <= 0.5:
                first_ratio_under_05 = blk
                note.append("first r<=0.5")
            ann = " <-- " + ", ".join(note) if note else ""
            print(f"  blk{blk:02d}: cos={cos:.4f} ratio_amax(eng/cli)={r:.4g}{ann}")
        print(f"\nFirst block with cos<0.95: {first_cos_under_95}")
        print(f"First block with ratio_amax>=2 (eng>cli): {first_ratio_over_2}")
        print(f"First block with ratio_amax<=0.5 (eng<cli, half or less): {first_ratio_under_05}")

if __name__ == "__main__":
    main()
