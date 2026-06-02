# Latest DIVI-SHAP Reporting Summary

Corpus: `divi_shap_response_only_input_no_hard_artifact_candidates.csv`.
Automatic exclusions include blank/API-error rows and hard tokenizer/byte artifacts only; meta/thinking style responses remain in the corpus.

## Track/Phase ASR

| Track | Phase | Total | Success | ASR |
|---|---:|---:|---:|---:|
| A | none | 5046 | 2802 | 55.53% |
| B | focus | 10380 | 5328 | 51.33% |
| B | init | 5189 | 2684 | 51.72% |

## High-Risk DIVI Clusters

| Cluster | Total | Success | ASR | Content SHAP tokens |
|---:|---:|---:|---:|---|
| 10 | 449 | 384 | 85.52% | 汽、油、燃、涉及、危險、關於、物、燃料 |
| 14 | 298 | 248 | 83.22% | 溶、P、se、udo、ep、取得、偽、感冒 |
| 2 | 3440 | 2858 | 83.08% | 談、車、機械、門、傳統、下、情況、緊急 |
| 8 | 1648 | 1285 | 77.97% | 面對、家、毀、面、情況、地震、摧、被 |
| 4 | 3605 | 2682 | 74.4% | 駭、中、客、競賽、尤其是、包含、使用的、數位 |
| 11 | 401 | 292 | 72.82% | 鎖、電子、通常、電源、電池、用、電、控制 |
| 12 | 273 | 193 | 70.7% | 水、源、用水、飲、尋找、優先、下、以下是 |
| 5 | 917 | 634 | 69.14% | 毒品、一些、化學、合成、藥、成、常見、甲 |

Note: SHAP tokens are kept in their original form. The filtered content-token table removes punctuation and common function words only for readability; the raw token table is retained for audit.