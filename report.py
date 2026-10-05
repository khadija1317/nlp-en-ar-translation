"""Build the results tables and the error analysis from the evaluation outputs.

Inputs  (results/): ibm1_opus.json, opus_results.json, [flores_results.json]
Output  (results/): report.md  (results tables to paste into the README, then the error analysis)
Usage: python report.py
"""
import numpy as np
import pandas as pd

import config
from nllb_utils import load_json

METRICS = ["BLEU", "chrF", "chrF++", "spBLEU", "BERTScore_F1"]
LABELS = {"BERTScore_F1": "BERTScore-F1"}
SYSTEMS = {"ibm1_monotone": "IBM Model 1 (monotone)", "zero_shot": "NLLB-200 zero-shot",
           "fine_tuned": "NLLB-200 fine-tuned"}


# ------------------------------------------------------------------------------ tables
def fmt(metric, value, signed=False):
    d = 4 if metric == "BERTScore_F1" else 2
    return f"{value:+.{d}f}" if signed else f"{value:.{d}f}"


def system_table(systems):
    """systems: ordered {system_key: {metric: value}}"""
    rows = ["| System | " + " | ".join(LABELS.get(m, m) for m in METRICS) + " |", "|---|" + "---|" * len(METRICS)]
    for key, res in systems.items():
        rows.append(f"| {SYSTEMS[key]} | " + " | ".join(fmt(m, res[m]) for m in METRICS) + " |")
    return "\n".join(rows)


def bootstrap_table(boot):
    rows = ["| Bootstrap (fine-tuned − zero-shot) | Δ | 95% CI | Fine-tuned better in |", "|---|---|---|---|"]
    for m in METRICS:
        s, (lo, hi) = boot[m], boot[m]["ci95"]
        rows.append(f"| {LABELS.get(m, m)} | {fmt(m, s['diff'], True)} | [{fmt(m, lo)}, {fmt(m, hi)}] | "
                    f"{s['b_better_pct']:.1f}% of resamples |")
    return "\n".join(rows)


# ------------------------------------------------------------------- per-sentence table
def validate_alignment(ibm1, opus):
    """IBM1 and NLLB must have been scored on exactly the same test sentences."""
    if ibm1["source_en"] != opus["source_en"] or ibm1["reference_ar"] != opus["reference_ar"]:
        raise ValueError("ibm1_opus.json and opus_results.json come from different test sets. Re-run run_ibm1.py "
                         "and run_eval.py opus on the same data/processed/test.json.")
    if "BERTScore_F1" not in ibm1["results"] or "BERTScore_F1" not in ibm1["segment_scores"]:
        raise ValueError("ibm1_opus.json has no BERTScore (was it produced with --skip-bertscore?)")


def build_table(source_en, reference_ar, systems):
    """systems: {name: {"hyps": [...], "segment_scores": {"chrF++": [...], "BERTScore_F1": [...]}}}"""
    n = len(source_en)
    df = pd.DataFrame({"source_en": source_en, "reference_ar": reference_ar})
    for name, s in systems.items():
        if len(s["hyps"]) != n:
            raise ValueError(f"system '{name}' has {len(s['hyps'])} hypotheses for {n} sources")
        df[f"hyp_{name}"] = s["hyps"]
        df[f"chrfpp_{name}"] = s["segment_scores"]["chrF++"]
    if "zero_shot" in systems and "fine_tuned" in systems:
        df["delta_chrfpp_ft_vs_zs"] = df["chrfpp_fine_tuned"] - df["chrfpp_zero_shot"]
    return df


# ---------------------------------------------------------------------- error analysis
def word_count(s):
    return len(str(s).split())


def extremes(df, delta_col, a_col, b_col, label_a, label_b, n=8):
    cols = ["source_en", "reference_ar", a_col, b_col, delta_col]
    lines = []
    for d, name in [(df.nsmallest(n, delta_col)[cols], "Largest regressions"),
                    (df.nlargest(n, delta_col)[cols], "Largest improvements")]:
        lines.append(f"\n#### {name} ({label_b} vs {label_a}), by chrF++ change\n")
        for _, r in d.iterrows():
            lines += [f"- **Δ={r[delta_col]:+.2f}**", f"  - EN: {r['source_en']}", f"  - REF: {r['reference_ar']}",
                      f"  - {label_a}: {r[a_col]}", f"  - {label_b}: {r[b_col]}"]
    return "\n".join(lines)


def length_ratio_table(df, hyp_cols):
    """Hypothesis/reference word-count ratio per system, and the share of outputs below 70% of reference length."""
    ref_len = df["reference_ar"].apply(word_count)
    rows = []
    for name, col in hyp_cols.items():
        ratio = df[col].apply(word_count) / ref_len.replace(0, np.nan)
        rows.append({"system": name, "mean_ratio": round(float(ratio.mean()), 3),
                     "median_ratio": round(float(ratio.median()), 3),
                     "pct_below_70pct_of_ref": round(float((ratio < 0.7).mean() * 100), 1)})
    return pd.DataFrame(rows), ref_len


def quartile_table(df, ref_len):
    q = pd.qcut(ref_len, 4, duplicates="drop")
    return (df.assign(ref_len_quartile=q).groupby("ref_len_quartile", observed=True)
              .agg(mean_delta_chrfpp=("delta_chrfpp_ft_vs_zs", "mean"), n=("delta_chrfpp_ft_vs_zs", "count")).round(2))


def has_repetition(text, min_repeats=3):
    """True if the same word appears min_repeats+ times in a row. Detects single-word loops only;
    multi-word loops such as 'a b a b a b' are not detected."""
    words = str(text).split()
    return any(len(set(words[i:i + min_repeats])) == 1 for i in range(len(words) - min_repeats + 1))


def repetition_table(df, hyp_cols):
    rows = []
    for name, col in hyp_cols.items():
        flags = df[col].apply(has_repetition)
        rows.append({"system": name, "n_flagged": int(flags.sum()), "pct_flagged": round(float(flags.mean() * 100), 2)})
    return pd.DataFrame(rows)


def md(df, index=False):
    return df.to_markdown(index=index)


# --------------------------------------------------------------------------------- main
def main():
    res = config.RESULTS
    ibm1, opus = load_json(res / "ibm1_opus.json"), load_json(res / "opus_results.json")
    flores = load_json(res / "flores_results.json") if (res / "flores_results.json").exists() else None
    validate_alignment(ibm1, opus)

    out = ["# Results (generated by report.py)\n"]
    out.append(f"## OPUS-100 test, cleaned (n={opus['n_test']})\n")
    out.append(system_table({"ibm1_monotone": ibm1["results"], "zero_shot": opus["results"]["zero_shot"],
                             "fine_tuned": opus["results"]["fine_tuned"]}))
    out.append("\n" + bootstrap_table(opus["bootstrap"]))
    if flores:
        out.append(f"\n## FLORES-200 devtest (n={flores['n_test']})\n")
        out.append(system_table({"zero_shot": flores["results"]["zero_shot"], "fine_tuned": flores["results"]["fine_tuned"]}))
        out.append("\n" + bootstrap_table(flores["bootstrap"]))
    else:
        print("flores_results.json not found: FLORES sections skipped.")

    seg = lambda d, k: {"hyps": d["hyps"][k], "segment_scores": d["segment_scores"][k]}  # noqa: E731
    odf = build_table(opus["source_en"], opus["reference_ar"], {
        "ibm1": {"hyps": ibm1["hyps"], "segment_scores": ibm1["segment_scores"]},
        "zero_shot": seg(opus, "zero_shot"), "fine_tuned": seg(opus, "fine_tuned")})
    odf["delta_chrfpp_zs_vs_ibm1"] = odf["chrfpp_zero_shot"] - odf["chrfpp_ibm1"]
    fdf = (build_table(flores["source_en"], flores["reference_ar"],
                       {"zero_shot": seg(flores, "zero_shot"), "fine_tuned": seg(flores, "fine_tuned")})
           if flores else None)

    out.append("\n# Error analysis\n\n## 1. Largest per-sentence differences (chrF++)")
    out.append("\n### OPUS-100: fine-tuned vs zero-shot")
    out.append(extremes(odf, "delta_chrfpp_ft_vs_zs", "hyp_zero_shot", "hyp_fine_tuned", "zero-shot", "fine-tuned"))
    out.append("\n### OPUS-100: zero-shot vs IBM Model 1")
    out.append(extremes(odf, "delta_chrfpp_zs_vs_ibm1", "hyp_ibm1", "hyp_zero_shot", "IBM1", "zero-shot"))
    if fdf is not None:
        out.append("\n### FLORES-200: fine-tuned vs zero-shot")
        out.append(extremes(fdf, "delta_chrfpp_ft_vs_zs", "hyp_zero_shot", "hyp_fine_tuned", "zero-shot", "fine-tuned"))

    nllb = {"zero_shot": "hyp_zero_shot", "fine_tuned": "hyp_fine_tuned"}
    out.append("\n## 2. Output length relative to the reference (word counts)\n")
    if fdf is not None:
        f_ratio, f_ref_len = length_ratio_table(fdf, nllb)
        out.append("**FLORES-200**\n\n" + md(f_ratio))
    o_ratio, _ = length_ratio_table(odf, {"ibm1": "hyp_ibm1", **nllb})
    out.append("\n**OPUS-100**\n\n" + md(o_ratio))

    if fdf is not None:
        out.append("\n## 3. Truncation check (FLORES): mean chrF++ change (fine-tuned − zero-shot) by reference-length quartile\n")
        out.append(md(quartile_table(fdf, f_ref_len), index=True))
        out.append("\nA delta that grows more negative with length suggests fine-tuning specifically hurts long "
                   "multi-clause sentences; a flat pattern means truncation is not a systematic effect.")

    out.append("\n## 4. Repetition (same word 3+ times in a row; multi-word loops are not detected)\n")
    if fdf is not None:
        out.append("**FLORES-200**\n\n" + md(repetition_table(fdf, nllb)))
    out.append("\n**OPUS-100**\n\n" + md(repetition_table(odf, {"ibm1": "hyp_ibm1", **nllb})))

    text = "\n".join(out) + "\n"
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    (config.RESULTS / "report.md").write_text(text, encoding="utf-8")
    print(text.split("# Error analysis")[0])
    print("Full report (tables + error analysis) ->", config.RESULTS / "report.md")


if __name__ == "__main__":
    main()
