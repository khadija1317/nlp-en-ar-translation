"""Shared code: IO, cleaning, sampling, IBM Model 1, NLLB translation, metrics, paired bootstrap."""
import json
import os
import random
import re
from pathlib import Path

import numpy as np
import sacrebleu
from sacrebleu.metrics import BLEU, CHRF

import config

# ----------------------------------------------------------------------------- IO


def read_lines(path):
    """One entry per line, trailing newline removed (nothing else is touched)."""
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def load_raw_split(raw_dir, en_name, ar_name):
    en, ar = read_lines(Path(raw_dir) / en_name), read_lines(Path(raw_dir) / ar_name)
    if len(en) != len(ar):
        raise ValueError(f"Misaligned split: {en_name} has {len(en)} lines, {ar_name} has {len(ar)}")
    return en, ar


def save_json(obj, path):
    """Atomic write, so an interrupted run never leaves a corrupt file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_pairs(path, en, ar):
    if len(en) != len(ar):
        raise ValueError("en/ar length mismatch")
    save_json({"en": en, "ar": ar}, path)


def load_pairs(path):
    d = load_json(path)
    if len(d["en"]) != len(d["ar"]):
        raise ValueError(f"{path}: en/ar length mismatch")
    return d["en"], d["ar"]


# ------------------------------------------------------------------------- cleaning
# Order: (1) strip, drop empty + duplicate pairs  (2) length/ratio filter  (3) strip subtitle tags, drop
# newly-empty pairs  (4) drop Arabic-side Latin contamination (exact copy of the English line; Latin letters
# but no Arabic script; mixed lines with a Latin-word share >= threshold).

TAG_PATTERN = re.compile(r"\{\\[^}]*\}")
LATIN_PATTERN = re.compile(r"[A-Za-z]")
ARABIC_PATTERN = re.compile(r"[\u0600-\u06FF]")


def word_lengths(lines):
    return np.array([len(line.split()) for line in lines], dtype=np.int64)


def latin_word_ratio(text):
    words = text.split()
    if not words:
        return 0.0
    return sum(1 for w in words if LATIN_PATTERN.search(w)) / len(words)


def strip_tags(line):
    return TAG_PATTERN.sub("", line).strip()


def dedup_and_strip(en_lines, ar_lines):
    seen, en_out, ar_out = set(), [], []
    n_empty = n_dupe = 0
    for en, ar in zip(en_lines, ar_lines):
        en_s, ar_s = en.strip(), ar.strip()
        if not en_s or not ar_s:
            n_empty += 1
            continue
        if (en_s, ar_s) in seen:
            n_dupe += 1
            continue
        seen.add((en_s, ar_s))
        en_out.append(en_s)
        ar_out.append(ar_s)
    return en_out, ar_out, n_empty, n_dupe


def length_ratio_filter(en_lines, ar_lines, max_len, max_ratio):
    en_lens, ar_lens = word_lengths(en_lines), word_lengths(ar_lines)
    ratio = np.maximum(en_lens, ar_lens) / np.maximum(np.minimum(en_lens, ar_lens), 1)
    keep = (en_lens <= max_len) & (ar_lens <= max_len) & (ratio <= max_ratio)
    return [e for e, k in zip(en_lines, keep) if k], [a for a, k in zip(ar_lines, keep) if k]


def contamination_drop_set(en_lines, ar_lines, latin_ratio_threshold):
    contaminated = [i for i, ar in enumerate(ar_lines) if LATIN_PATTERN.search(ar)]
    exact = {i for i in contaminated if en_lines[i].strip() == ar_lines[i].strip()}
    mixed = [i for i in contaminated if i not in exact and ARABIC_PATTERN.search(ar_lines[i])]
    other = {i for i in contaminated if i not in exact and i not in set(mixed)}
    high = {i for i in mixed if latin_word_ratio(ar_lines[i]) >= latin_ratio_threshold}
    drop = exact | other | high
    return drop, {"n_arabic_with_latin": len(contaminated), "n_exact_copy": len(exact),
                  "n_latin_only_not_copy": len(other), "n_mixed_high_latin_ratio": len(high),
                  "n_contamination_dropped": len(drop)}


def clean_parallel(en_lines, ar_lines, max_len=None, max_ratio=None, latin_ratio=None):
    """Full cleaning pipeline. Returns (en, ar, stats)."""
    max_len = config.MAX_LEN if max_len is None else max_len
    max_ratio = config.MAX_RATIO if max_ratio is None else max_ratio
    latin_ratio = config.LATIN_RATIO_THRESHOLD if latin_ratio is None else latin_ratio
    if len(en_lines) != len(ar_lines):
        raise ValueError("en/ar length mismatch")
    stats = {"n_input": len(en_lines)}

    en, ar, n_empty, n_dupe = dedup_and_strip(en_lines, ar_lines)
    stats.update(n_empty=n_empty, n_duplicate=n_dupe, n_after_dedup=len(en))

    en, ar = length_ratio_filter(en, ar, max_len, max_ratio)
    stats["n_after_length_ratio"] = len(en)
    stats["n_removed_length_ratio"] = stats["n_after_dedup"] - len(en)

    stats["n_tag_lines_en"] = sum(1 for x in en if TAG_PATTERN.search(x))
    stats["n_tag_lines_ar"] = sum(1 for x in ar if TAG_PATTERN.search(x))
    en, ar = [strip_tags(x) for x in en], [strip_tags(x) for x in ar]
    kept = [(e, a) for e, a in zip(en, ar) if e and a]
    stats["n_empty_after_tag_strip"] = len(en) - len(kept)
    en, ar = [p[0] for p in kept], [p[1] for p in kept]
    stats["n_after_tag_strip"] = len(en)

    drop, breakdown = contamination_drop_set(en, ar, latin_ratio)
    stats.update(breakdown)
    keep_idx = [i for i in range(len(en)) if i not in drop]
    en, ar = [en[i] for i in keep_idx], [ar[i] for i in keep_idx]
    stats["n_final"] = len(en)
    stats["n_removed_total"] = stats["n_input"] - len(en)
    return en, ar, stats


def sample_pairs(en_lines, ar_lines, n, seed):
    """Seeded subsample: shuffle all indices, take the first n (same procedure as the original run)."""
    if len(en_lines) != len(ar_lines):
        raise ValueError("en/ar length mismatch")
    if n > len(en_lines):
        raise ValueError(f"Cannot sample {n} pairs from a pool of {len(en_lines)}")
    idx = list(range(len(en_lines)))
    random.Random(seed).shuffle(idx)        # same stream as random.seed(seed); random.shuffle(idx)
    idx = idx[:n]
    return [en_lines[i] for i in idx], [ar_lines[i] for i in idx]


def overlap_report(train_en, train_ar, eval_en, eval_ar):
    """Exact-match overlap of an eval split with training data (report only; nothing is removed)."""
    sources, pairs = set(train_en), set(zip(train_en, train_ar))
    n = len(eval_en)
    src = sum(1 for e in eval_en if e in sources)
    pair = sum(1 for e, a in zip(eval_en, eval_ar) if (e, a) in pairs)
    return {"n_eval": n, "source_in_train": src, "pair_in_train": pair,
            "source_in_train_pct": round(100 * src / n, 2) if n else 0.0,
            "pair_in_train_pct": round(100 * pair / n, 2) if n else 0.0}


# ---------------------------------------------------------------------- IBM Model 1
# NLTK convention: for AlignedSent(words, mots), translation_table[w][m] = P(w | m), with an extra `None`
# (NULL word) as the second key. To get P(arabic | english), Arabic goes in `words` and English in `mots`.


def train_ibm1(en_lines, ar_lines, iterations=5):
    from nltk.translate import AlignedSent, IBMModel1
    bitext = [AlignedSent(ar.split(), en.split()) for en, ar in zip(en_lines, ar_lines)]
    return IBMModel1(bitext, iterations)


def best_translations(model):
    """English word -> Arabic word with the highest P(arabic | english). Ties break alphabetically."""
    best, best_p = {}, {}
    for ar_word, row in model.translation_table.items():
        if ar_word is None:
            continue
        for en_word, p in row.items():
            if en_word is None:
                continue
            cur = best_p.get(en_word)
            if cur is None or p > cur or (p == cur and ar_word < best[en_word]):
                best_p[en_word], best[en_word] = p, ar_word
    return best


def top_translations(model, words, k=5):
    wanted = set(words)
    cands = {w: [] for w in wanted}
    for ar_word, row in model.translation_table.items():
        if ar_word is None:
            continue
        for en_word, p in row.items():
            if en_word in wanted:
                cands[en_word].append((ar_word, p))
    return {w: sorted(c, key=lambda x: (-x[1], x[0]))[:k] for w, c in cands.items()}


def translate_monotone(sentence, best, oov_passthrough=True):
    out = []
    for w in sentence.split():
        if w in best:
            out.append(best[w])
        elif oov_passthrough:
            out.append(w)           # unseen English word is copied through unchanged
    return " ".join(out)


# ------------------------------------------------------------------------------ NLLB
# heavy imports are inside the functions so the CPU-only scripts do not need torch/transformers


def hf_token():
    return os.environ.get("HF_TOKEN") or None


def get_device(allow_cpu=False):
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if not allow_cpu:
        raise RuntimeError("No GPU found. Enable one (Kaggle: Settings > Accelerator) or pass --allow-cpu (very slow).")
    return "cpu"


def load_tokenizer(model_name, src_lang, tgt_lang=None):
    from transformers import AutoTokenizer
    kw = {"src_lang": src_lang, "token": hf_token()}
    if tgt_lang:
        kw["tgt_lang"] = tgt_lang
    return AutoTokenizer.from_pretrained(model_name, **kw)


def load_model(name_or_path, device):
    from transformers import AutoModelForSeq2SeqLM
    return AutoModelForSeq2SeqLM.from_pretrained(name_or_path, token=hf_token()).to(device)


def translate(model, tokenizer, sentences, num_beams, batch_size, max_length, desc=None):
    import torch
    from tqdm.auto import tqdm
    model.eval()
    tokenizer.src_lang = config.SRC_LANG
    tgt_id = tokenizer.convert_tokens_to_ids(config.TGT_LANG)
    out = []
    with torch.no_grad():
        for i in tqdm(range(0, len(sentences), batch_size), desc=desc):
            enc = tokenizer(sentences[i:i + batch_size], return_tensors="pt", padding=True,
                            truncation=True, max_length=max_length).to(model.device)
            gen = model.generate(**enc, forced_bos_token_id=tgt_id, max_new_tokens=max_length,
                                 num_beams=num_beams, do_sample=False)
            out.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
    return out


# ---------------------------------------------------------------------------- metrics
_BERTSCORE = None


def score_full(hyps, refs, bertscore=True):
    """Corpus BLEU / chrF / chrF++ / spBLEU / BERTScore-F1, plus per-sentence chrF++ and BERTScore."""
    global _BERTSCORE
    if len(hyps) != len(refs):
        raise ValueError(f"{len(hyps)} hypotheses vs {len(refs)} references")
    chrf, chrfpp = CHRF(), CHRF(word_order=2)
    corpus = {
        "BLEU": round(sacrebleu.corpus_bleu(hyps, [refs]).score, 2),
        "chrF": round(chrf.corpus_score(hyps, [refs]).score, 2),
        "chrF++": round(chrfpp.corpus_score(hyps, [refs]).score, 2),
        "spBLEU": round(sacrebleu.corpus_bleu(hyps, [refs], tokenize="flores200").score, 2),
    }
    seg = {"chrF++": [round(chrfpp.sentence_score(h, [r]).score, 2) for h, r in zip(hyps, refs)]}
    if bertscore:
        if _BERTSCORE is None:
            import evaluate                     # Hugging Face `evaluate` library
            _BERTSCORE = evaluate.load("bertscore")
        bs = _BERTSCORE.compute(predictions=hyps, references=refs, lang=config.BERTSCORE_LANG)
        corpus["BERTScore_F1"] = round(float(np.mean(bs["f1"])), 4)
        seg["BERTScore_F1"] = [round(float(x), 4) for x in bs["f1"]]
    return corpus, seg


# --------------------------------------------------------------------------- bootstrap
# Paired bootstrap (Koehn, 2004), B vs A. Corpus metrics are resampled through sacrebleu's per-segment
# sufficient statistics, so each resample is a true corpus-level score; BERTScore on its per-sentence mean.


def _summ(diff_obs, diffs):
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": round(float(diff_obs), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "b_better_pct": round(float((diffs > 0).mean() * 100), 1)}


def boot_corpus_metric(metric, hyp_a, hyp_b, refs, n_boot=1000, seed=42):
    sa = np.array(metric._extract_corpus_statistics(hyp_a, [refs]))
    sb = np.array(metric._extract_corpus_statistics(hyp_b, [refs]))
    n = len(refs)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for k in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[k] = (metric._compute_score_from_stats(sb[idx].sum(0)).score
                    - metric._compute_score_from_stats(sa[idx].sum(0)).score)
    obs = metric._compute_score_from_stats(sb.sum(0)).score - metric._compute_score_from_stats(sa.sum(0)).score
    public = metric.corpus_score(hyp_b, [refs]).score - metric.corpus_score(hyp_a, [refs]).score
    if abs(obs - public) > 1e-6:      # guards against sacrebleu internals changing
        raise RuntimeError(f"Bootstrap statistics disagree with sacrebleu corpus scores ({obs} vs {public}); "
                           "check the installed sacrebleu version.")
    return _summ(obs, diffs)


def boot_mean(a, b, n_boot=1000, seed=42):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n_boot, len(a)))
    return _summ(b.mean() - a.mean(), b[idx].mean(1) - a[idx].mean(1))


def paired_bootstrap(hyp_a, hyp_b, refs, seg_a, seg_b, n_boot=1000, seed=42):
    """{metric: {diff, ci95, b_better_pct}} for B - A (A = zero-shot, B = fine-tuned)."""
    out = {
        "BLEU": boot_corpus_metric(BLEU(), hyp_a, hyp_b, refs, n_boot, seed),
        "chrF": boot_corpus_metric(CHRF(), hyp_a, hyp_b, refs, n_boot, seed),
        "chrF++": boot_corpus_metric(CHRF(word_order=2), hyp_a, hyp_b, refs, n_boot, seed),
        "spBLEU": boot_corpus_metric(BLEU(tokenize="flores200"), hyp_a, hyp_b, refs, n_boot, seed),
    }
    out["BERTScore_F1"] = boot_mean(seg_a["BERTScore_F1"], seg_b["BERTScore_F1"], n_boot, seed)
    return out
