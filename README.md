# Does fine-tuning NLLB-200 on noisy OPUS-100 help or hurt English→Arabic translation?

Fine-tuning a strong multilingual translation model on a noisy, subtitle-heavy corpus usually improves the score on that corpus's own test set. The question this project asks is whether that reflects *better translation* or only *adaptation to the corpus*. To separate the two, NLLB-200-distilled-600M is evaluated zero-shot and after fine-tuning on an in-domain test set (OPUS-100) and an out-of-domain one (FLORES-200), with paired bootstrap significance tests and five metrics chosen for Arabic morphology.

**Main finding.** On the out-of-domain FLORES-200 set, fine-tuning significantly *reduces* quality on all five metrics (for example −2.09 BLEU, 95% CI [−2.73, −1.49]), even though it is trained for in-domain gains. In-domain, on the cleaned OPUS-100 test set, the same fine-tuning significantly *improves* all five metrics (for example +2.40 BLEU, 95% CI [1.72, 3.12]). Evaluating only on held-out data from the fine-tuning distribution would have hidden this and mistaken domain adaptation for general improvement. In-domain results are in [§5.1](#51-in-domain-opus-100-test-set).

The conclusion is specific to this setup: a 600M-parameter model, a 40k-pair subsample, a single run, greedy decoding (see [Limitations](#7-limitations)).

---

## 1. Systems compared

| System | Description |
|---|---|
| IBM Model 1 | Statistical baseline trained from scratch on the 40k sample (NLTK EM, 5 iterations). Monotone word-by-word decoding: each English word is replaced by the Arabic word with the highest P(Arabic \| English); unseen words are copied unchanged. |
| NLLB-200 zero-shot | `facebook/nllb-200-distilled-600M`, no training. |
| NLLB-200 fine-tuned | The same model fine-tuned on the 40k-pair OPUS-100 sample. |

All neural decoding is greedy (`num_beams=1`) with a 128-token limit, identical across systems and test sets.

## 2. Data

**OPUS-100 English–Arabic** (Zhang et al., 2020): about 1,000,000 training pairs and 2,000-pair dev and test splits. It is a heterogeneous web-sourced corpus, mostly informal subtitle text with a small share of formal (UN-style) documents, plus markup remnants, untranslated lines, and duplicates.

**FLORES-200 devtest** (NLLB Team, 2022; 1,012 sentences) is used only for out-of-domain evaluation: clean, professionally translated text in a more formal register.

### Cleaning

One function (`nllb_en_ar.data.clean`) is applied identically to train, dev and test, in this order:

1. Strip whitespace; drop pairs with an empty side and exact duplicate pairs.
2. Drop pairs with more than 100 words on either side, or a length ratio above 4 (larger word count ÷ smaller).
3. Strip leftover subtitle-formatting tags (such as `{\cH62D0DD}`) and drop pairs that become empty.
4. Drop Arabic-side contamination: exact copies of the English line, lines with Latin letters and no Arabic script, and mixed lines in which at least half of the words are Latin.

The length and ratio thresholds were chosen by inspecting how many pairs each cutoff would remove, not from a principled statistical criterion. On the training split:

| Stage | Pairs remaining |
|---|---:|
| Raw OPUS-100 en-ar train | 1,000,000 |
| After empty/duplicate removal | 959,929 |
| After length/ratio filter | 949,014 |
| After tag stripping and empty removal | 948,984 |
| After contamination filter (cleaned pool) | 941,277 |
| Random sample used for IBM1 and fine-tuning (seed 42) | 40,000 |

When run on the standard OPUS-100 file, `prepare_data.py` checks these counts, and the first five sampled sentences, against the run that trained the evaluated checkpoint, and exits with an error if they differ. Dev goes from 2,000 to 1,963 pairs and test from 2,000 to 1,961. An exact-match overlap check between the cleaned dev/test sets and the 40k training sample found 0 identical source sentences and 0 identical pairs (the check only reports; it removes nothing). Full per-split counts are in `data/processed/cleaning_report.json`.

Because the test set is cleaned, its absolute scores are **not comparable** with numbers reported on the raw OPUS-100 test split elsewhere.

## 3. Fine-tuning

| Setting | Value |
|---|---|
| Base model | `facebook/nllb-200-distilled-600M`, EN `eng_Latn` → AR `arb_Arab` |
| Learning rate / schedule | 3e-5, linear decay, 500 warmup steps |
| Batch size | 2 per device × 8 gradient-accumulation steps = 16 effective |
| Epochs (maximum) | 3 |
| Optimizer / precision | 8-bit AdamW (`bitsandbytes`), weight decay 0.01, fp16, gradient checkpointing |
| Max sequence length | 128 tokens |
| Evaluation / early stopping | dev loss every 250 steps, patience 3 evaluations |
| Seed | 42 |

Training stopped early at step 5,750 (epoch 2.3 of 3). Dev loss fell from 2.150 (step 250) to its minimum of 1.713 at step 5,000, and the three evaluations after that did not improve on it. The weights from the best step were restored before saving. The run took about 3.2 hours on a Kaggle GPU. The evaluated checkpoint used a dev set of 1,999 pairs (one corrupted 171,144-word line removed, nothing else); `finetune.py` uses the fully cleaned dev set, so a fresh run would not reproduce that checkpoint exactly.

## 4. Evaluation

Five metrics, chosen to cover complementary notions of quality for a morphologically rich language:

- **BLEU** (Papineni et al., 2002), computed with sacreBLEU (Post, 2018): word n-gram precision. Reported for comparability, but it penalizes valid alternative Arabic word forms (clitics, prefixes) as errors.
- **chrF** (Popović, 2015): character n-gram F-score, far more tolerant of Arabic morphology.
- **chrF++** (Popović, 2017): chrF plus word unigrams and bigrams. **Primary metric** for interpretation.
- **spBLEU** (Goyal et al., 2022): BLEU on the FLORES-200 SentencePiece tokenization instead of whitespace tokens.
- **BERTScore-F1** (Zhang et al., 2020) with `bert-base-multilingual-cased`: contextual-embedding similarity. Its scale is compressed, so only gaps between systems are interpreted, and cautiously, since mBERT's Arabic representation is only moderate.

Significance: paired bootstrap resampling (Koehn, 2004), 1,000 resamples, seed 42, 95% intervals, for fine-tuned minus zero-shot. Corpus metrics are resampled through sacreBLEU's per-segment statistics so each resample is a true corpus-level score.

## 5. Results

### 5.1 In-domain: OPUS-100 test set

Cleaned OPUS-100 test split (n = 1,961 pairs; see [Cleaning](#cleaning)).

| System | BLEU | chrF | chrF++ | spBLEU | BERTScore-F1 |
|---|---|---|---|---|---|
| IBM Model 1 (monotone) | 4.16 | 26.00 | 22.92 | 7.19 | 0.7281 |
| NLLB-200 zero-shot | 15.08 | 44.99 | 40.59 | 24.53 | 0.8162 |
| NLLB-200 fine-tuned | 17.48 | 46.23 | 41.86 | 26.73 | 0.8183 |

| Bootstrap (fine-tuned − zero-shot) | Δ | 95% CI | Fine-tuned better in |
|---|---|---|---|
| BLEU | +2.40 | [1.72, 3.12] | 100.0% of resamples |
| chrF | +1.25 | [0.72, 1.78] | 100.0% of resamples |
| chrF++ | +1.27 | [0.77, 1.79] | 100.0% of resamples |
| spBLEU | +2.19 | [1.57, 2.85] | 100.0% of resamples |
| BERTScore-F1 | +0.0021 | [0.0004, 0.0038] | 99.2% of resamples |

Fine-tuning significantly improves all five metrics in-domain, with every confidence interval excluding zero. The BERTScore gain is the smallest and its interval the closest to zero (fine-tuned better in 99.2% of resamples). Both neural systems are far above the IBM Model 1 baseline.

### 5.2 Out-of-domain: FLORES-200 devtest (n = 1,012)

| System | BLEU | chrF | chrF++ | spBLEU | BERTScore-F1 |
|---|---|---|---|---|---|
| NLLB-200 zero-shot | 21.98 | 53.87 | 50.23 | 33.96 | 0.8612 |
| NLLB-200 fine-tuned | 19.88 | 52.17 | 48.27 | 31.80 | 0.8512 |

| Bootstrap (fine-tuned − zero-shot) | Δ | 95% CI | Fine-tuned better in |
|---|---|---|---|
| BLEU | -2.09 | [-2.73, -1.49] | 0.0% of resamples |
| chrF | -1.69 | [-2.18, -1.22] | 0.0% of resamples |
| chrF++ | -1.96 | [-2.44, -1.49] | 0.0% of resamples |
| spBLEU | -2.17 | [-2.76, -1.61] | 0.0% of resamples |
| BERTScore-F1 | -0.0100 | [-0.0120, -0.0079] | 0.0% of resamples |

All five metrics show a statistically significant decline under fine-tuning, with every confidence interval excluding zero. The out-of-domain loss (−2.09 BLEU) is of similar size to the in-domain gain (+2.40 BLEU).

## 6. Error analysis

Per-sentence chrF++ differences were computed for every comparison, together with automatic checks for two candidate failure modes. `report.py` regenerates all of it, including the best and worst individual sentences for both test sets, in `results/report.md`.

**Truncation (FLORES).** The mean output/reference word-count ratio is nearly identical for zero-shot and fine-tuned (0.981 vs 0.973), but the share of outputs shorter than 70% of the reference rises from 2.9% to 4.4%. The mean chrF++ change across reference-length quartiles (−1.90, −1.60, −2.13, −2.13) shows no clean trend with length, so truncation is not concentrated in long sentences: several of the largest regressions are outputs cut off after the first clause, including on short sentences.

**Repetition (FLORES).** 4 of 1,012 fine-tuned outputs (0.40%) and 1 zero-shot output (0.10%) contain the same word three or more times in a row; one of the largest fine-tuned regressions is a visible loop on a single word. The check detects single-word loops only and misses multi-word ones: another large regression repeats a whole phrase twice and is not flagged.

**FLORES: a redistribution of failure modes, not uniform degradation.** Fine-tuning fixes some of zero-shot's most severe failures (outputs unrelated to the source sentence; a long repetition loop) but introduces its own: truncated outputs and occasional repetition. The net effect over the full set is a significant decline, consistent with specialization toward the training domain rather than a general loss of translation ability.

**OPUS-100.** The share of fine-tuned and zero-shot outputs with a repeated word is similar (0.71% vs 0.61%, 14 vs 12 outputs). The largest per-sentence swings in both directions are short subtitle-style utterances, where changing one word moves chrF++ a lot. Several of the largest regressions replace an output that already matched the reference with a different, plausible wording or inflection, while some of the largest gains reproduce corpus conventions, for example the UN-style date rendering "تموز/يوليه 2009" and the fuller "البند 19 من جدول الأعمال" where zero-shot gave "يوليو 2009" and "البند 19". These are examples chosen for their size, not a representative sample, and single-reference metrics can penalize valid alternative translations; the full lists, including the zero-shot vs IBM Model 1 comparison, are in `results/report.md`.

## 7. Limitations

**Data.**
- OPUS-100's Arabic is nominally MSA, but no check was made for register or dialect consistency with FLORES-200; informal or dialectal leakage in the subtitle text may account for part of the domain gap.
- The cleaning thresholds are heuristic, and comparable outliers could remain undetected in the ~941,000-pair pool.
- Cleaning the test set removes some hard-but-legitimate lines (for example proper-noun-only or untranslatable lines), so the cleaned test set may be somewhat easier than the raw one.

**Fine-tuning.**
- Only 40,000 pairs (under 5% of the cleaned pool) were sampled, once, without stratification by length, genre or register. The size was set by compute, not justified statistically.
- One run, one seed, one hyperparameter configuration, no tuning sweep. The batch size, 8-bit optimizer and gradient checkpointing were chosen to fit available hardware.
- Checkpoint selection used dev loss on OPUS-100 only, so nothing in training rewarded keeping out-of-domain ability. This is a plausible contributor to the FLORES decline, not a demonstrated cause.
- The aggregate `train_loss` in the final trainer output (15.02) is inconsistent with the per-step training losses (about 1.6–2.5) and the dev loss (about 1.7). It is very likely a logging-scale artifact of gradient accumulation; the per-step values are the ones to read.
- Loading the fine-tuned checkpoint prints a weight-tying warning (`model.shared.weight`, `lm_head.weight` and the encoder/decoder embeddings are stored as separate tensors), which also explains its larger-than-expected file size. It was not root-caused and does not appear to affect scores, but this was not separately verified.

**Scope.** Only the smallest distilled NLLB-200 model (600M) and only English→Arabic were tested. Larger checkpoints may retain multilingual ability better under fine-tuning, so the specialization-versus-generalization finding should not be assumed to hold at larger scale.

**Evaluation.**
- All decoding is greedy; beam search was not evaluated.
- All conclusions rest on automatic, single-reference metrics; there is no COMET or human evaluation.
- BERTScore uses mBERT, whose Arabic quality is only moderate.
- Ten bootstrap tests (five metrics × two datasets) are reported without multiple-comparison correction; the consistent direction across all five metrics on each dataset makes joint chance results implausible.
- The truncation and repetition heuristics use whitespace word counts (not clitic-aware), and the repetition check misses multi-word loops.

**Reproducibility.** Fine-tuning and evaluation were run across several Kaggle sessions with differing package versions (pinned ranges are in `requirements.txt`; results were produced on `transformers` 4.x). Kaggle kernels that were reused between scripts once contaminated results, which is why each evaluation now runs as its own process. The fine-tuned weights are stored only on the Hugging Face Hub; no other backup was kept.

## 8. Reproducing

```bash
pip install -r requirements.txt

# 1. Data: download the en-ar archive from the OPUS-100 page (https://opus.nlpl.eu/OPUS-100), e.g.
#    https://object.pouta.csc.fi/OPUS-100/v1.0/opus-100-corpus-ar-en-v1.0.tar.gz
#    and put the six files opus.ar-en-{train,dev,test}.{en,ar} in data/raw/
python prepare_data.py        # clean all splits, sample 40k, check against the original run
python run_ibm1.py            # CPU: IBM Model 1 baseline
python finetune.py            # GPU: trains the model and saves it to models/nllb_final
python run_eval.py opus       # GPU: zero-shot vs fine-tuned + bootstrap (add --finetuned models/nllb_final for your own model)
python run_eval.py flores     # GPU: run in a fresh process / session
python report.py              # tables + error analysis -> results/report.md
```

All settings are in `config.py`. The Hugging Face token, if you need one, is read **only** from the `HF_TOKEN` environment variable (on Kaggle: add it as a secret and set `os.environ["HF_TOKEN"]` from `UserSecretsClient().get_secret("HF_TOKEN")`). Raw data, processed data and model weights are git-ignored.

`

## AI assistance

Claude was used throughout this project as a collaborative tool.

## References

- Brown, P. F., Della Pietra, S. A., Della Pietra, V. J., & Mercer, R. L. (1993). The Mathematics of Statistical Machine Translation: Parameter Estimation. *Computational Linguistics*, 19(2).
- Goyal, N., et al. (2022). The FLORES-101 Evaluation Benchmark for Low-Resource and Multilingual Machine Translation. *TACL*.
- Koehn, P. (2004). Statistical Significance Tests for Machine Translation Evaluation. *EMNLP 2004*.
- NLLB Team, et al. (2022). No Language Left Behind: Scaling Human-Centered Machine Translation. arXiv:2207.04672.
- Papineni, K., Roukos, S., Ward, T., & Zhu, W.-J. (2002). BLEU: a Method for Automatic Evaluation of Machine Translation. *ACL 2002*.
- Popović, M. (2015). chrF: character n-gram F-score for automatic MT evaluation. *WMT 2015*.
- Popović, M. (2017). chrF++: words helping character n-grams. *WMT 2017*.
- Post, M. (2018). A Call for Clarity in Reporting BLEU Scores. *WMT 2018*.
- Zhang, B., Williams, P., Titov, I., & Sennrich, R. (2020). Improving Massively Multilingual Neural Machine Translation and Zero-Shot Translation. *ACL 2020*. (introduces OPUS-100)
- Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., & Artzi, Y. (2020). BERTScore: Evaluating Text Generation with BERT. *ICLR 2020*.
