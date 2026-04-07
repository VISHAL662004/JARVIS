# Comparison With Existing Legal Analysis Baselines

| Model | Keypoint F1 | Judge Detect | Section Detect | Decision Capture | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|---:|---:|---:|---:|
| Your Pipeline (Scratch + Heuristic Fallback) | 0.8404 | 0.3925 | 0.6750 | 0.8000 | 0.4412 | 0.3634 | 0.3793 |
| Existing Rule-Based + TextRank-Style Baseline | 0.8707 | 0.3100 | 0.6750 | 0.8100 | 0.4102 | 0.3417 | 0.3584 |
| Existing Lead-3 Extractive Baseline | 0.4425 | 0.3100 | 0.6750 | 0.3450 | 0.1313 | 0.0696 | 0.0912 |

## Winner By Metric
- keypoint_f1_mean: Existing Rule-Based + TextRank-Style Baseline (0.8707)
- judge_detect_rate: Your Pipeline (Scratch + Heuristic Fallback) (0.3925)
- section_detect_rate: Your Pipeline (Scratch + Heuristic Fallback) (0.6750)
- decision_capture_rate: Existing Rule-Based + TextRank-Style Baseline (0.8100)
- rouge1: Your Pipeline (Scratch + Heuristic Fallback) (0.4412)
- rouge2: Your Pipeline (Scratch + Heuristic Fallback) (0.3634)
- rougeL: Your Pipeline (Scratch + Heuristic Fallback) (0.3793)