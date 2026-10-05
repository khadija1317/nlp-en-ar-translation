"""Evaluate zero-shot and fine-tuned NLLB on one test set, with paired bootstrap significance. Needs a GPU.

  python run_eval.py opus      # cleaned OPUS-100 test (data/processed/test.json)
  python run_eval.py flores    # FLORES-200 devtest (downloaded from the Hub)

Run each dataset in a FRESH process/session (a reused kernel once contaminated results).
Writes results/{opus,flores}_results.json
"""
import argparse
import gc
import time

import config
from nllb_utils import (get_device, load_model, load_pairs, load_tokenizer, paired_bootstrap, save_json,
                        score_full, translate)


def load_eval_set(dataset):
    if dataset == "opus":
        en, ar = load_pairs(config.DATA_PROCESSED / "test.json")
        return "opus100_test_cleaned", en, ar
    from datasets import load_dataset
    ds = load_dataset(config.FLORES_DATASET, split=config.FLORES_SPLIT)
    for col in ("eng_Latn", "arb_Arab"):
        if col not in ds.column_names:
            raise SystemExit(f"Column '{col}' missing from {config.FLORES_DATASET}; found {ds.column_names}")
    return "flores200_devtest", list(ds["eng_Latn"]), list(ds["arb_Arab"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=["opus", "flores"])
    ap.add_argument("--finetuned", default=config.FINETUNED_MODEL, help="Hub repo id or local path")
    ap.add_argument("--allow-cpu", action="store_true")
    args = ap.parse_args()

    import torch
    device = get_device(args.allow_cpu)
    name, src, refs = load_eval_set(args.dataset)
    n = len(src)
    assert len(refs) == n
    print(f"Device: {device} | {name}: {n} pairs")

    g = config.GENERATION
    systems = {"zero_shot": config.BASE_MODEL, "fine_tuned": args.finetuned}
    out_path = config.RESULTS / f"{args.dataset}_results.json"
    payload = {"dataset": name, "n_test": n, "source_en": src, "reference_ar": refs, "models": systems,
               "generation": g, "results": {}, "hyps": {}, "segment_scores": {}}

    tok = load_tokenizer(config.BASE_MODEL, config.SRC_LANG)      # same tokenizer for both systems
    for sys_name, path in systems.items():
        model = load_model(path, device)
        t0 = time.time()
        hyps = translate(model, tok, src, g["num_beams"], g["batch_size"], g["max_length"],
                         desc=f"{args.dataset}/{sys_name}")
        elapsed = time.time() - t0
        assert len(hyps) == n, f"{sys_name}: {len(hyps)} hypotheses for {n} sources"
        res, seg = score_full(hyps, refs)
        res["decode_seconds"] = round(elapsed, 1)
        payload["hyps"][sys_name], payload["results"][sys_name], payload["segment_scores"][sys_name] = hyps, res, seg
        print(sys_name, res)
        save_json(payload, out_path)                 # incremental, atomic
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    boot = paired_bootstrap(payload["hyps"]["zero_shot"], payload["hyps"]["fine_tuned"], refs,
                            payload["segment_scores"]["zero_shot"], payload["segment_scores"]["fine_tuned"],
                            n_boot=config.BOOTSTRAP_RESAMPLES, seed=config.BOOTSTRAP_SEED)
    payload["bootstrap"] = boot
    save_json(payload, out_path)
    print(f"\n=== {name}: fine_tuned - zero_shot, 95% CI ===")
    for k, s in boot.items():
        print(f"{k:13s} diff={s['diff']:+.4f}  CI95={s['ci95']}  fine-tuned better in {s['b_better_pct']}% of resamples")
    print("Saved ->", out_path)


if __name__ == "__main__":
    main()
