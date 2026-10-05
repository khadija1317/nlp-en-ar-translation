"""Train the IBM Model 1 baseline on the 40k sample and decode the cleaned OPUS-100 test set (CPU only).
Writes results/ibm1_opus.json.   Usage: python run_ibm1.py
"""
import argparse
import time

import config
from nllb_utils import (best_translations, load_pairs, save_json, score_full, top_translations,
                        train_ibm1, translate_monotone)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-bertscore", action="store_true", help="debug only; the output cannot be used by report.py")
    args = ap.parse_args()

    tr_en, tr_ar = load_pairs(config.DATA_PROCESSED / "train_sample.json")
    te_en, te_ar = load_pairs(config.DATA_PROCESSED / "test.json")
    print(f"train sample: {len(tr_en)} pairs | test: {len(te_en)} pairs")

    t0 = time.time()
    model = train_ibm1(tr_en, tr_ar, config.IBM1_ITERATIONS)
    print(f"IBM1 trained in {time.time() - t0:.1f}s")
    for w, tops in top_translations(model, ["I", "you", "the", "love", "know"], k=3).items():
        print(f"  {w!r}: " + ", ".join(f"{a} ({p:.3f})" for a, p in tops))

    best = best_translations(model)
    t0 = time.time()
    hyps = [translate_monotone(s, best) for s in te_en]
    elapsed = time.time() - t0
    corpus, seg = score_full(hyps, te_ar, bertscore=not args.skip_bertscore)
    corpus["decode_seconds"] = round(elapsed, 1)
    print(corpus)

    out = config.RESULTS / "ibm1_opus.json"
    save_json({"system": "ibm1_monotone", "n_test": len(te_en), "source_en": te_en, "reference_ar": te_ar,
               "results": corpus, "hyps": hyps, "segment_scores": seg}, out)
    print("Saved ->", out)


if __name__ == "__main__":
    main()
