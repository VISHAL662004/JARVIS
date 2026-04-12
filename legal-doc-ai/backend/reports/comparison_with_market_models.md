# Comparison With Actual Market Models

| Model | Keypoint F1 | Judge Detect | Section Detect | Decision Capture | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|---:|---:|---:|---:|
| Your Pipeline (Scratch) | 0.8564 | 0.5000 | 0.6667 | 0.6667 | 0.4293 | 0.3396 | 0.3585 |
| Market Stack B: typeform/distilbert-base-uncased-mnli + dbmdz/bert-large-cased-finetuned-conll03-english + sshleifer/distilbart-cnn-12-6 | 0.6246 | 0.2500 | 0.6667 | 0.8333 | 0.5023 | 0.4217 | 0.4673 |

## Winner By Metric
- keypoint_f1_mean: Your Pipeline (Scratch) (0.8564)
- judge_detect_rate: Your Pipeline (Scratch) (0.5000)
- section_detect_rate: Your Pipeline (Scratch) (0.6667)
- decision_capture_rate: Market Stack B: typeform/distilbert-base-uncased-mnli + dbmdz/bert-large-cased-finetuned-conll03-english + sshleifer/distilbart-cnn-12-6 (0.8333)
- rouge1: Market Stack B: typeform/distilbert-base-uncased-mnli + dbmdz/bert-large-cased-finetuned-conll03-english + sshleifer/distilbart-cnn-12-6 (0.5023)
- rouge2: Market Stack B: typeform/distilbert-base-uncased-mnli + dbmdz/bert-large-cased-finetuned-conll03-english + sshleifer/distilbart-cnn-12-6 (0.4217)
- rougeL: Market Stack B: typeform/distilbert-base-uncased-mnli + dbmdz/bert-large-cased-finetuned-conll03-english + sshleifer/distilbart-cnn-12-6 (0.4673)

## Separate My Model vs Each Market Model

### Source Stack: Market Stack B: typeform/distilbert-base-uncased-mnli + dbmdz/bert-large-cased-finetuned-conll03-english + sshleifer/distilbart-cnn-12-6

#### Your Pipeline vs typeform/distilbert-base-uncased-mnli (Zero-shot Classifier)
| Metric | Your Pipeline | Market Model | Winner |
|---|---:|---:|---|
| keypoint_f1_mean | 0.8564 | 0.6246 | Your Pipeline |
| decision_capture_rate | 0.6667 | 0.8333 | Market Model |

#### Your Pipeline vs dbmdz/bert-large-cased-finetuned-conll03-english (NER Model)
| Metric | Your Pipeline | Market Model | Winner |
|---|---:|---:|---|
| judge_detect_rate | 0.5000 | 0.2500 | Your Pipeline |
| section_detect_rate | 0.6667 | 0.6667 | Tie |

#### Your Pipeline vs sshleifer/distilbart-cnn-12-6 (Summarizer Model)
| Metric | Your Pipeline | Market Model | Winner |
|---|---:|---:|---|
| rouge1 | 0.4293 | 0.5023 | Market Model |
| rouge2 | 0.3396 | 0.4217 | Market Model |
| rougeL | 0.3585 | 0.4673 | Market Model |