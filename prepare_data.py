"""Raw OPUS-100 (ar-en) -> cleaned train pool -> seeded 40k train sample, plus cleaned dev and test.
The same cleaning is applied to train, dev and test.

Writes to data/processed/: train_sample.json, dev.json, test.json, cleaning_report.json
Usage: python prepare_data.py [--skip-verify]
"""
import argparse
import sys

import config
from nllb_utils import (clean_parallel, load_raw_split, overlap_report, sample_pairs, save_json, save_pairs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-verify", action="store_true",
                    help="do not compare against the run that trained the released checkpoint")
    args = ap.parse_args()

    cleaned, report = {}, {}
    for split in ("train", "dev", "test"):
        en, ar = load_raw_split(config.DATA_RAW, *config.RAW_FILES[split])
        en, ar, st = clean_parallel(en, ar)
        cleaned[split], report[split] = (en, ar), st
        print(f"[{split}] {st['n_input']} raw -> {st['n_final']} clean (duplicates {st['n_duplicate']}, "
              f"empty {st['n_empty']}, length/ratio {st['n_removed_length_ratio']}, "
              f"contamination {st['n_contamination_dropped']})")

    pool_en, pool_ar = cleaned["train"]
    s_en, s_ar = sample_pairs(pool_en, pool_ar, config.N_TRAIN, config.SEED)
    report["sample"] = {"n": config.N_TRAIN, "seed": config.SEED, "pool_size": len(pool_en)}
    print(f"[train] sampled {len(s_en)} pairs (seed={config.SEED}) from a pool of {len(pool_en)}")

    # check against the run that trained the released checkpoint (only meaningful on the standard file)
    exp, st = config.EXPECTED, report["train"]
    if args.skip_verify:
        report["verification"] = "skipped"
    elif st["n_input"] != exp["n_input"]:
        print(f"Verification not applicable: raw train has {st['n_input']} lines, reference run had {exp['n_input']}.")
        report["verification"] = "not_applicable"
    else:
        problems = [f"{k}: got {st[k]}, expected {exp[k]}"
                    for k in ("n_after_dedup", "n_after_length_ratio", "n_after_tag_strip", "n_final")
                    if st[k] != exp[k]]
        for i, want in enumerate(exp["sample_head_en"]):
            got = s_en[i] if i < len(s_en) else "<missing>"
            if got != want:
                problems.append(f"sample[{i}]: got {got!r}, expected {want!r}")
        if problems:
            print("\nVERIFICATION FAILED: this does not reproduce the training data of the released checkpoint:")
            print("\n".join("  - " + p for p in problems))
            sys.exit(1)
        print("Verification passed: pool counts and the head of the 40k sample match the original run.")
        report["verification"] = "passed"

    report["overlap_with_40k_sample"] = {}
    for split in ("dev", "test"):
        o = overlap_report(s_en, s_ar, *cleaned[split])
        report["overlap_with_40k_sample"][split] = o
        print(f"[{split}] vs 40k sample: {o['source_in_train']} identical sources, {o['pair_in_train']} identical pairs")

    out = config.DATA_PROCESSED
    save_pairs(out / "train_sample.json", s_en, s_ar)
    save_pairs(out / "dev.json", *cleaned["dev"])
    save_pairs(out / "test.json", *cleaned["test"])
    save_json(report, out / "cleaning_report.json")
    print(f"\nWrote train_sample.json, dev.json, test.json, cleaning_report.json to {out}")


if __name__ == "__main__":
    main()
