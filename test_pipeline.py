"""Tests (CPU only, no model downloads). Run with:  pytest"""
import itertools
import json
import random
import runpy
import sys

import numpy as np
import pytest
import sacrebleu
from sacrebleu.metrics import BLEU, CHRF

import config
import nllb_utils as U
import report


@pytest.fixture(autouse=True)
def offline_spbleu(monkeypatch):
    """The 'flores200' tokenizer needs a model download; substitute 13a in tests only."""
    orig = sacrebleu.corpus_bleu

    def corpus_bleu(h, r, **kw):
        if kw.get("tokenize") == "flores200":
            kw["tokenize"] = "13a"
        return orig(h, r, **kw)

    def bleu(*a, **kw):
        if kw.get("tokenize") == "flores200":
            kw["tokenize"] = "13a"
        return BLEU(*a, **kw)

    monkeypatch.setattr(sacrebleu, "corpus_bleu", corpus_bleu)
    monkeypatch.setattr(U, "BLEU", bleu)


W = lambda n: " ".join(["w"] * n)  # noqa: E731


# ------------------------------------------------------------------------ cleaning
def test_dedup_strip_empty():
    en, ar, n_empty, n_dupe = U.dedup_and_strip(["a ", "a", "", "b", "c", "d", "d"], ["x", "x ", "y", "z", " ", "p", "q"])
    assert (en, ar) == (["a", "b", "d", "d"], ["x", "z", "p", "q"])   # dedup is on the pair, not the side
    assert (n_empty, n_dupe) == (2, 1)


def test_length_ratio_boundaries():
    en = [W(100), W(101), W(8), W(9), W(1), W(5)]
    ar = [W(100), W(50), W(2), W(2), W(4), W(1)]
    e, a = U.length_ratio_filter(en, ar, 100, 4)    # 101 words, ratios 4.5 and 5 dropped; 100 words and ratio 4 kept
    assert [len(x.split()) for x in e] == [100, 8, 1] and [len(x.split()) for x in a] == [100, 2, 4]


def test_tags_and_contamination_rules():
    assert U.strip_tags(r"{\cH62D0DD}hello {\pos(1,2)}") == "hello"
    en = ["Hello there", "Mr Smith came", "Nothing", "2001", "Half done", "pure arabic", "Buy now please"]
    ar = ["Hello there",             # Latin + exact copy               -> drop
          "جاء Mr Smith",             # Arabic+Latin, Latin share 2/3    -> drop
          "Untranslated text",        # Latin only, not a copy           -> drop
          "2001",                     # no Latin, exact copy             -> KEEP
          "ذهب إلى Paris غدا",        # Arabic+Latin, share 1/4          -> keep
          "عربي صافي",                # pure Arabic                      -> keep
          "اشتر Buy الآن Now"]        # Arabic+Latin, share 2/4 = 0.5    -> drop (>= threshold)
    drop, br = U.contamination_drop_set(en, ar, 0.5)
    assert drop == {0, 1, 2, 6} and br["n_exact_copy"] == 1 and br["n_arabic_with_latin"] == 5


def test_pipeline_counts_and_dev_style_outlier():
    en = ["a b", "a b", "ok one", W(150), "hello", "fine pair", r"tag", "short line", "x y"]
    ar = ["ا ب", "ا ب", "حسنا واحد", W(150), "hello", "زوج جيد", r"{\an8}", "سطر", " ".join(["كلمة"] * 5000)]
    e, a, st = U.clean_parallel(en, ar)
    assert st["n_duplicate"] == 1 and st["n_after_dedup"] == 8
    assert st["n_after_length_ratio"] == 6            # the 150-word pair and the 5000-word alignment failure are gone
    assert st["n_empty_after_tag_strip"] == 1 and st["n_contamination_dropped"] == 1
    assert st["n_final"] == len(e) == len(a) == 4
    e2, a2, st2 = U.clean_parallel(e, a)               # idempotent
    assert (e, a) == (e2, a2) and st2["n_removed_total"] == 0


def test_sample_matches_original_procedure():
    en = [f"e{i}" for i in range(1000)]
    ar = [f"a{i}" for i in range(1000)]
    random.seed(42)                                    # what the notebook did
    idx = list(range(1000))
    random.shuffle(idx)
    s_en, s_ar = U.sample_pairs(en, ar, 50, 42)
    assert s_en == [en[i] for i in idx[:50]] and [x[1:] for x in s_en] == [x[1:] for x in s_ar]
    with pytest.raises(ValueError):
        U.sample_pairs(["a"], ["b"], 2, 1)


def test_overlap_report():
    r = U.overlap_report(["a", "b"], ["x", "y"], ["a", "b", "c"], ["x", "WRONG", "z"])
    assert (r["n_eval"], r["source_in_train"], r["pair_in_train"]) == (3, 2, 1)


# --------------------------------------------------------------------------- IBM1
EN = ["I", "you", "see", "love", "dog", "cat"]
AR = ["انا", "انت", "ارى", "احب", "كلب", "قطة"]


def ibm1_corpus():
    en, ar = [], []
    for idx in itertools.permutations(range(6), 3):
        en.append(" ".join(EN[i] for i in idx))
        ar.append(" ".join(AR[i] for i in idx))
    return en, ar


def test_ibm1_direction_is_p_arabic_given_english():
    en, ar = ibm1_corpus()
    en.append("I")
    ar.append("نادر")                  # a rare Arabic token that only ever co-occurs with "I"
    best = U.best_translations(U.train_ibm1(en, ar, 10))
    assert best["dog"] == "كلب" and best["cat"] == "قطة" and best["love"] == "احب"
    # the inverted direction P(en|ar) (the original notebook's behaviour) would pick the rare token (p=1.0)
    assert best["I"] == "انا" and None not in best and None not in best.values()


def test_ibm1_top_translations_form_a_distribution_over_arabic():
    en, ar = ibm1_corpus()
    model = U.train_ibm1(en, ar, 10)
    allp = U.top_translations(model, ["dog"], k=10 ** 6)["dog"]
    assert allp[0][0] == "كلب" and abs(sum(p for _, p in allp) - 1.0) < 1e-3


def test_translate_monotone_oov():
    assert U.translate_monotone("the dog", {"dog": "كلب"}) == "the كلب"
    assert U.translate_monotone("the dog", {"dog": "كلب"}, oov_passthrough=False) == "كلب"


# ------------------------------------------------------------ metrics and bootstrap
REFS = [f"the quick brown fox number {i} jumps over lazy dogs again and again" for i in range(60)]
GOOD = list(REFS)
BAD = [r.replace("quick", "slow").replace("lazy", "tired") for r in REFS]


def test_score_full_without_bertscore():
    corpus, seg = U.score_full(GOOD, REFS, bertscore=False)
    assert set(corpus) == {"BLEU", "chrF", "chrF++", "spBLEU"} and corpus["BLEU"] == 100.0
    assert len(seg["chrF++"]) == 60
    with pytest.raises(ValueError):
        U.score_full(["a"], ["a", "b"], bertscore=False)


@pytest.mark.parametrize("metric", [BLEU(), CHRF(), CHRF(word_order=2)])
def test_bootstrap(metric):
    up = U.boot_corpus_metric(metric, BAD, GOOD, REFS, n_boot=200, seed=1)       # B better than A
    assert up["diff"] > 0 and up["ci95"][0] > 0 and up["b_better_pct"] == 100.0
    same = U.boot_corpus_metric(metric, BAD, BAD, REFS, n_boot=200, seed=1)
    assert same["diff"] == 0 and same["ci95"] == [0.0, 0.0]
    down = U.boot_corpus_metric(metric, GOOD, BAD, REFS, n_boot=200, seed=1)
    assert down["diff"] < 0 and down["b_better_pct"] == 0.0
    assert U.boot_corpus_metric(metric, BAD, GOOD, REFS, 100, 7) == U.boot_corpus_metric(metric, BAD, GOOD, REFS, 100, 7)


def test_boot_mean_and_paired_keys():
    s = U.boot_mean(np.full(50, 0.70), np.full(50, 0.75), n_boot=100, seed=0)
    assert s["diff"] == 0.05 and s["b_better_pct"] == 100.0
    out = U.paired_bootstrap(BAD, GOOD, REFS, {"BERTScore_F1": [0.7] * 60}, {"BERTScore_F1": [0.8] * 60}, 50, 0)
    assert list(out) == ["BLEU", "chrF", "chrF++", "spBLEU", "BERTScore_F1"]


# ----------------------------------------------------------------------------- report
def test_report_tables_and_helpers():
    sysm = {"zero_shot": {"BLEU": 14.97, "chrF": 44.35, "chrF++": 40.03, "spBLEU": 24.41, "BERTScore_F1": 0.8144}}
    assert "| NLLB-200 zero-shot | 14.97 | 44.35 | 40.03 | 24.41 | 0.8144 |" in report.system_table(sysm)
    boot = {m: {"diff": -0.73, "ci95": [-1.21, -0.26], "b_better_pct": 0.1} for m in report.METRICS}
    boot["BERTScore_F1"] = {"diff": 0.0028, "ci95": [0.0011, 0.0045], "b_better_pct": 100.0}
    t = report.bootstrap_table(boot)
    assert "| chrF | -0.73 | [-1.21, -0.26] | 0.1% of resamples |" in t
    assert "| BERTScore-F1 | +0.0028 | [0.0011, 0.0045] | 100.0% of resamples |" in t
    assert report.has_repetition("رعاية رعاية رعاية") and report.has_repetition("x y z z z")
    assert not report.has_repetition("a b a b a b") and not report.has_repetition("a a b b c c")
    assert not report.has_repetition("one two")


def test_report_alignment_and_table():
    src, ref = ["s0", "s1"], ["r0", "r1"]
    seg = {"chrF++": [10.0, 10.0], "BERTScore_F1": [0.7, 0.7]}
    ibm1 = {"source_en": src, "reference_ar": ref, "segment_scores": seg, "results": {"BERTScore_F1": 0.7}}
    opus = {"source_en": src, "reference_ar": ref}
    report.validate_alignment(ibm1, opus)
    with pytest.raises(ValueError):
        report.validate_alignment(ibm1, {**opus, "source_en": src[::-1]})
    with pytest.raises(ValueError):
        report.validate_alignment({**ibm1, "segment_scores": {"chrF++": [1, 1]}, "results": {}}, opus)
    df = report.build_table(src, ref, {"zero_shot": {"hyps": ["z", "z"], "segment_scores": seg},
                                       "fine_tuned": {"hyps": ["f", "f"], "segment_scores": {"chrF++": [12.0, 12.0], "BERTScore_F1": [0.8, 0.8]}}})
    assert (df["delta_chrfpp_ft_vs_zs"] == 2.0).all()


# ------------------------------------------------------- end to end (CPU, synthetic)
def run_script(name, *args):
    old, sys.argv = sys.argv, [name, *args]
    try:
        runpy.run_path(str(config.ROOT / name), run_name="__main__")
        return 0
    except SystemExit as e:
        return e.code or 0
    finally:
        sys.argv = old


def make_split(n, offset=0, outlier=False):
    en = [f"item{i} is on the table {i % 5}" for i in range(offset, offset + n)]
    ar = [f"عنصر{i} على الطاولة {i % 5}" for i in range(offset, offset + n)]
    en += ["duplicate line here", "duplicate line here", "Just English", "tagged"]
    ar += ["سطر مكرر هنا", "سطر مكرر هنا", "Just English", r"{\an8}موسوم"]
    if outlier:
        en.append("normal english line")
        ar.append(" ".join(["كلمة"] * 800))
    return en, ar


@pytest.fixture()
def project(tmp_path, monkeypatch):
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    for split, (n, off, outl) in {"train": (150, 0, False), "dev": (15, 1000, True), "test": (20, 2000, True)}.items():
        en, ar = make_split(n, off, outl)
        en_name, ar_name = config.RAW_FILES[split]
        (raw / en_name).write_text("\n".join(en) + "\n", encoding="utf-8")
        (raw / ar_name).write_text("\n".join(ar) + "\n", encoding="utf-8")
    monkeypatch.setattr(config, "DATA_RAW", raw)
    monkeypatch.setattr(config, "DATA_PROCESSED", tmp_path / "data" / "processed")
    monkeypatch.setattr(config, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(config, "N_TRAIN", 60)
    return tmp_path


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def test_full_cpu_pipeline(project):
    proc, res = config.DATA_PROCESSED, config.RESULTS
    assert run_script("prepare_data.py") == 0
    rep = read(proc / "cleaning_report.json")
    assert rep["verification"] == "not_applicable"
    for split in ("train", "dev", "test"):                      # cleaning reached every split
        assert rep[split]["n_duplicate"] == 1 and rep[split]["n_contamination_dropped"] == 1
        assert rep[split]["n_tag_lines_ar"] == 1
    assert rep["dev"]["n_removed_length_ratio"] == 1 and rep["test"]["n_removed_length_ratio"] == 1
    test = read(proc / "test.json")
    assert len(test["en"]) == len(test["ar"]) == rep["test"]["n_final"] == 22
    assert all(len(a.split()) <= 100 for a in test["ar"]) and "موسوم" in test["ar"]
    assert len(read(proc / "train_sample.json")["en"]) == 60
    first = (proc / "train_sample.json").read_text(encoding="utf-8")
    assert run_script("prepare_data.py") == 0 and (proc / "train_sample.json").read_text(encoding="utf-8") == first

    assert run_script("run_ibm1.py", "--skip-bertscore") == 0
    ibm1 = read(res / "ibm1_opus.json")
    assert ibm1["source_en"] == test["en"] and ibm1["reference_ar"] == test["ar"] and len(ibm1["hyps"]) == 22

    # report.py needs BERTScore and NLLB outputs (GPU/models): fabricate them
    n = ibm1["n_test"]
    ibm1["results"]["BERTScore_F1"], ibm1["segment_scores"]["BERTScore_F1"] = 0.7, [0.7] * n
    (res / "ibm1_opus.json").write_text(json.dumps(ibm1, ensure_ascii=False), encoding="utf-8")

    def fake(name, src, ref):
        r = {k: {"BLEU": 10.0, "chrF": 40.0, "chrF++": 35.0, "spBLEU": 20.0, "BERTScore_F1": 0.8} for k in ("zero_shot", "fine_tuned")}
        sg = lambda c: {"chrF++": [c] * len(src), "BERTScore_F1": [0.8] * len(src)}  # noqa: E731
        b = {m: {"diff": 1.0, "ci95": [0.5, 1.5], "b_better_pct": 99.0} for m in report.METRICS}
        (res / f"{name}_results.json").write_text(json.dumps({
            "n_test": len(src), "source_en": src, "reference_ar": ref, "results": r,
            "hyps": {"zero_shot": list(ref), "fine_tuned": [x.replace("الطاولة", "طاولة") for x in ref]},
            "segment_scores": {"zero_shot": sg(30.0), "fine_tuned": sg(31.0)}, "bootstrap": b}, ensure_ascii=False), encoding="utf-8")

    fake("opus", test["en"], test["ar"])
    assert run_script("report.py") == 0                          # FLORES optional: skipped
    text = (res / "report.md").read_text(encoding="utf-8")
    assert "IBM Model 1 (monotone)" in text and "Repetition" in text and "FLORES-200 devtest" not in text
    fake("flores", [f"f{i} sentence" for i in range(12)], [f"جملة {i}" for i in range(12)])
    assert run_script("report.py") == 0
    assert "FLORES-200 devtest (n=12)" in (res / "report.md").read_text(encoding="utf-8")


def test_verification_fails_then_passes(project, monkeypatch):
    bad = {"n_input": 154, "n_after_dedup": 1, "n_after_length_ratio": 1, "n_after_tag_strip": 1, "n_final": 1,
           "sample_head_en": ["nope"]}
    monkeypatch.setattr(config, "EXPECTED", bad)
    assert run_script("prepare_data.py") == 1                    # standard-size file but wrong counts
    assert run_script("prepare_data.py", "--skip-verify") == 0
    st = read(config.DATA_PROCESSED / "cleaning_report.json")["train"]
    head = read(config.DATA_PROCESSED / "train_sample.json")["en"][:5]
    monkeypatch.setattr(config, "EXPECTED", {**{k: st[k] for k in bad if k != "sample_head_en"}, "sample_head_en": head})
    assert run_script("prepare_data.py") == 0
    assert read(config.DATA_PROCESSED / "cleaning_report.json")["verification"] == "passed"


def test_misaligned_raw_files_fail_loudly(project):
    p = config.DATA_RAW / config.RAW_FILES["test"][0]
    p.write_text(p.read_text(encoding="utf-8") + "extra line\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Misaligned"):
        run_script("prepare_data.py")
