# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Techmanthan  
**Team Members:** Sahil Sahu, Udit Tiwari, Anjali Lodhi  
**College:** Sagar Institute of Science, Technology and Engineering  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

This submission uses a scalable two-stage entity-resolution system: deterministic
multi-key blocking generates a small plausible candidate set, then a
gradient-boosted binary classifier scores each candidate pair.  All processing
uses only the supplied TSV files.  The pipeline is disk-backed with DuckDB so it
can process the approximately 19 million supplied records on an 8 GB-RAM
laptop.

## 2. Methodology

### 2.1 Problem Analysis

The same business appears with punctuation changes, legal-suffix changes,
abbreviations, word reordering, partial/missing names, typos, transliterated
text, and partial/reordered addresses.  Postal codes are often absent, so they
are useful but cannot be the only retrieval rule.  In training, only 5.58% of
Source-1 records are true singletons; most have multiple links, so the model
predicts candidate pairs independently rather than imposing a one-to-one match
constraint.

### 2.2 Solution Strategy

**Approach Type:** Multi-key blocking plus supervised gradient-boosted pair
classifier.

**Core innovation:** The retrieval stage combines exact high-precision keys with
rare token, prefix/suffix, character-gram, and address-number keys. Candidate
ranking is frequency-aware: evidence from a rare shared key is prioritised over
evidence from a generic key. This retains candidates through noisy spelling and
word-order transformations while block-frequency and top-k caps control volume.

## 3. Candidate Generation (Blocking)

Before blocking, names and addresses are lower-cased, punctuation-normalized,
and compacted.  Business names additionally receive a legal-suffix-free form.
The following country-aware blocking keys are emitted:

- exact compact business name;
- compact legal-suffix-free business name;
- exact compact address;
- postal code plus first name token, and postal code plus compact-name prefix;
- leading address number plus compact-name prefix;
- rare full name tokens and rare address tokens;
- first/last three-character name-token keys;
- four-character compact-name grams; and
- four-character address-token prefixes; and
- country-scoped address-number tokens, which remain stable when address words
  are reordered.

Keys with more than 100 target records are excluded. Candidate evidence is
weighted by rule specificity and inverse target-block frequency, then
aggregated; the final model receives at most 50 records per Source-1 business.
The improved test candidate run produced 54,153,678 candidate pairs for
1,732,544 Source-1 records (32.079 candidates on average among non-empty
candidate lists). The matching validation candidate configuration recovered
70.6298% of true links while retaining a mean 29.274 candidates among
non-empty validation lists.

## 4. Matching Model

**Features used:**

- **Name features:** Jaro-Winkler similarity on normalized and suffix-free
  names, normalized Levenshtein/edit similarity, token-set Jaccard overlap,
  length ratio, exact-name flags, and first-token equality.
- **Address features:** Jaro-Winkler similarity, normalized edit similarity,
  token-set Jaccard overlap, shared numeric-token count, length ratio,
  exact-address flag, postal-code equality, and leading-address-number equality.
- **Other:** aggregated blocking score, strongest blocking-rule weight, and a
  Source-2 versus Source-3 indicator.

**Model type:** scikit-learn `HistGradientBoostingClassifier`.  It was trained
on 707,887 sampled positive candidate pairs and 1,114,706 sampled negative
candidate pairs from four Source-1 hash folds.  The fifth fold was held out.

**Threshold selection method:** Candidate probabilities were evaluated on the
untouched fold using the challenge's macro F_0.5 definition, including
singletons. A local fine search selected a conservative threshold of **0.940**.

## 5. Results & Error Analysis

- **Best validation macro F_0.5:** **0.800554** at threshold 0.940 on the
  expanded candidate validation experiment.
- **Internal pair-level checks:** Baseline ROC-AUC 0.99406 and average
  precision 0.99120; the final model adds order-invariant token-overlap
  features for robustness to address reordering.
- **Common false positives:** generic business terms that share an address
  fragment or a common name token; a high threshold limits these false merges.
- **Common false negatives:** records whose business name is missing or has a
  heavy corruption, and records with no surviving shared blocking key.  Such
  links cannot be recovered after candidate generation.

## 6. Conclusion

The solution balances candidate-set size, recall, and the precision-heavy
F_0.5 objective using scalable local processing.  It emits both the exact
candidate set consumed by the model and final high-confidence matches, with one
output row for every test Source-1 record.

## Appendix

### A. Code Artefacts

The `src/` directory contains data inspection, normalization/database build,
ground-truth conversion, blocking, candidate generation, training feature
generation, model training, batch scoring, threshold evaluation, and
partitioned final export scripts.  `README.md` provides the complete command
sequence to reproduce both TSV files.  `requirements.txt` pins dependencies.

### B. Additional Results

The code also includes `build_address_number_blocks.py` for the optional
address-number retrieval index and `validate_large_submission.py` for
structural validation on an 8 GB RAM machine. Final test output counts are
generated directly from the final scoring/export command and validated before
packaging.
