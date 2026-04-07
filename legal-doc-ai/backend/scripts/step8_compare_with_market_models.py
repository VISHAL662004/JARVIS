#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import re
import sys

import numpy as np
from rouge_score import rouge_scorer
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.services.entity_extractor import EntityExtractionService
from app.services.keypoint_extractor import KeyPointExtractionService
from app.services.summarizer import SummarizationService
from app.utils.text import remove_boilerplate, sentence_split


def load_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out


FINAL_PAT = re.compile(r"\b(allowed|dismissed|ordered|directed|granted|modified|released)\b", re.IGNORECASE)
SECTION_PAT = re.compile(r"\b(?:Section|Sec\.|Article)\s+[0-9A-Za-z()/-]+", re.IGNORECASE)


@dataclass
class EvalRow:
    name: str
    keypoint_f1_mean: float
    judge_detect_rate: float
    section_detect_rate: float
    decision_capture_rate: float
    rouge1: float
    rouge2: float
    rougeL: float

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "keypoint_f1_mean": self.keypoint_f1_mean,
            "judge_detect_rate": self.judge_detect_rate,
            "section_detect_rate": self.section_detect_rate,
            "decision_capture_rate": self.decision_capture_rate,
            "rouge1": self.rouge1,
            "rouge2": self.rouge2,
            "rougeL": self.rougeL,
        }


def doc_level_f1(gold: set[str], pred: set[str]) -> float:
    inter = len(gold & pred)
    precision = inter / max(1, len(pred))
    recall = inter / max(1, len(gold))
    return 2 * precision * recall / max(1e-8, precision + recall)


def markdown_table(rows: list[EvalRow]) -> str:
    header = (
        "| Model | Keypoint F1 | Judge Detect | Section Detect | Decision Capture | ROUGE-1 | ROUGE-2 | ROUGE-L |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|"
    )
    lines = [header]
    for r in rows:
        lines.append(
            f"| {r.name} | {r.keypoint_f1_mean:.4f} | {r.judge_detect_rate:.4f} | {r.section_detect_rate:.4f} | "
            f"{r.decision_capture_rate:.4f} | {r.rouge1:.4f} | {r.rouge2:.4f} | {r.rougeL:.4f} |"
        )
    return "\n".join(lines)


def eval_pipeline_model(
    key_test: list[dict],
    sum_test: list[dict],
    key_doc_limit: int,
    summary_doc_limit: int,
) -> EvalRow:
    ent = EntityExtractionService()
    key = KeyPointExtractionService()
    summ = SummarizationService()

    grouped: dict[str, list[dict]] = {}
    for row in key_test:
        grouped.setdefault(row["doc_id"], []).append(row)

    key_f1, judge, section, decision = [], [], [], []
    for _, rows in tqdm(list(grouped.items())[:key_doc_limit], desc="Your pipeline", unit="doc"):
        text = " ".join(r["sentence"] for r in rows)
        pred_labels = {p.label for p in key.extract(text)}
        gold_labels = {r["label"] for r in rows}
        key_f1.append(doc_level_f1(gold_labels, pred_labels))

        ext = ent.extract(text)
        judge.append(float(len(ext.judges) > 0))
        section.append(float(len(ext.legal_sections_cited) > 0))
        decision.append(float(ext.final_order is not None and bool(FINAL_PAT.search(ext.final_order.value))))

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_vals = []
    for row in tqdm(sum_test[:summary_doc_limit], desc="Your summarizer", unit="doc"):
        pred = summ.summarize_abstractive(remove_boilerplate(row["source"]))
        rouge_vals.append(scorer.score(row["target"], pred))

    return EvalRow(
        name="Your Pipeline (Scratch)",
        keypoint_f1_mean=float(np.mean(key_f1)) if key_f1 else 0.0,
        judge_detect_rate=float(np.mean(judge)) if judge else 0.0,
        section_detect_rate=float(np.mean(section)) if section else 0.0,
        decision_capture_rate=float(np.mean(decision)) if decision else 0.0,
        rouge1=float(np.mean([x["rouge1"].fmeasure for x in rouge_vals])) if rouge_vals else 0.0,
        rouge2=float(np.mean([x["rouge2"].fmeasure for x in rouge_vals])) if rouge_vals else 0.0,
        rougeL=float(np.mean([x["rougeL"].fmeasure for x in rouge_vals])) if rouge_vals else 0.0,
    )


def _safe_device() -> int:
    try:
        import torch

        return 0 if torch.cuda.is_available() else -1
    except Exception:
        return -1


def eval_market_stack(
    key_test: list[dict],
    sum_test: list[dict],
    name: str,
    zero_shot_model: str,
    ner_model: str,
    summarizer_model: str,
    key_doc_limit: int,
    summary_doc_limit: int,
) -> EvalRow:
    from transformers import pipeline

    device = _safe_device()
    zsc = pipeline("zero-shot-classification", model=zero_shot_model, device=device)
    ner = pipeline("ner", model=ner_model, aggregation_strategy="simple", device=device)
    summ = pipeline("summarization", model=summarizer_model, device=device)

    labels = ["FACT", "ISSUE", "ARGUMENT", "REASONING", "DECISION"]

    grouped: dict[str, list[dict]] = {}
    for row in key_test:
        grouped.setdefault(row["doc_id"], []).append(row)

    key_f1, judge, section, decision = [], [], [], []
    for _, rows in tqdm(list(grouped.items())[:key_doc_limit], desc=f"{name} keypoints", unit="doc"):
        doc_text = " ".join(r["sentence"] for r in rows)
        sents = sentence_split(remove_boilerplate(doc_text))[:25]

        pred_labels = set()
        for s in sents:
            try:
                out = zsc(s, labels, multi_label=False)
                if out["scores"][0] >= 0.50:
                    pred_labels.add(str(out["labels"][0]).upper())
            except Exception:
                continue

        gold_labels = {r["label"] for r in rows}
        key_f1.append(doc_level_f1(gold_labels, pred_labels))

        judge_found = False
        try:
            head = doc_text[:1800]
            ner_out = ner(head)
            for ent in ner_out:
                word = str(ent.get("word", ""))
                entity_group = str(ent.get("entity_group", "")).upper()
                if entity_group in {"PER", "PERSON"} and len(word.replace("##", "").split()) >= 2:
                    judge_found = True
                    break
        except Exception:
            judge_found = False

        judge.append(float(judge_found))
        section.append(float(bool(SECTION_PAT.search(doc_text))))
        decision.append(float("DECISION" in pred_labels or bool(FINAL_PAT.search(doc_text[-1500:]))))

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_vals = []
    for row in tqdm(sum_test[:summary_doc_limit], desc=f"{name} summary", unit="doc"):
        source = remove_boilerplate(row["source"])[:2800]
        pred = ""
        try:
            out = summ(source, max_length=120, min_length=30, do_sample=False, truncation=True)
            if out and isinstance(out, list):
                pred = str(out[0].get("summary_text", ""))
        except Exception:
            pred = ""
        rouge_vals.append(scorer.score(row["target"], pred))

    return EvalRow(
        name=name,
        keypoint_f1_mean=float(np.mean(key_f1)) if key_f1 else 0.0,
        judge_detect_rate=float(np.mean(judge)) if judge else 0.0,
        section_detect_rate=float(np.mean(section)) if section else 0.0,
        decision_capture_rate=float(np.mean(decision)) if decision else 0.0,
        rouge1=float(np.mean([x["rouge1"].fmeasure for x in rouge_vals])) if rouge_vals else 0.0,
        rouge2=float(np.mean([x["rouge2"].fmeasure for x in rouge_vals])) if rouge_vals else 0.0,
        rougeL=float(np.mean([x["rougeL"].fmeasure for x in rouge_vals])) if rouge_vals else 0.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare with named market models")
    parser.add_argument("--split-dir", type=Path, default=Path("data/processed/splits"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--key-doc-limit", type=int, default=120)
    parser.add_argument("--summary-doc-limit", type=int, default=120)
    parser.add_argument(
        "--run-stacks",
        type=str,
        default="B",
        choices=["A", "B", "all"],
        help="Which market stack(s) to evaluate. Use B for faster runs.",
    )
    args = parser.parse_args()

    key_test = load_jsonl(args.split_dir / "keypoints_test.jsonl")
    sum_test = load_jsonl(args.split_dir / "summary_test.jsonl")

    rows: list[EvalRow] = [
        eval_pipeline_model(
            key_test,
            sum_test,
            key_doc_limit=args.key_doc_limit,
            summary_doc_limit=args.summary_doc_limit,
        )
    ]

    if args.run_stacks in {"A", "all"}:
        rows.append(
            eval_market_stack(
                key_test=key_test,
                sum_test=sum_test,
                name="Market Stack A: facebook/bart-large-mnli + dslim/bert-base-NER + facebook/bart-large-cnn",
                zero_shot_model="facebook/bart-large-mnli",
                ner_model="dslim/bert-base-NER",
                summarizer_model="facebook/bart-large-cnn",
                key_doc_limit=args.key_doc_limit,
                summary_doc_limit=args.summary_doc_limit,
            )
        )

    if args.run_stacks in {"B", "all"}:
        rows.append(
            eval_market_stack(
                key_test=key_test,
                sum_test=sum_test,
                name="Market Stack B: typeform/distilbert-base-uncased-mnli + dbmdz/bert-large-cased-finetuned-conll03-english + sshleifer/distilbart-cnn-12-6",
                zero_shot_model="typeform/distilbert-base-uncased-mnli",
                ner_model="dbmdz/bert-large-cased-finetuned-conll03-english",
                summarizer_model="sshleifer/distilbart-cnn-12-6",
                key_doc_limit=args.key_doc_limit,
                summary_doc_limit=args.summary_doc_limit,
            )
        )

    args.report_dir.mkdir(parents=True, exist_ok=True)
    out_json = {
        "rows": [r.as_dict() for r in rows],
        "config": {
            "key_doc_limit": args.key_doc_limit,
            "summary_doc_limit": args.summary_doc_limit,
            "run_stacks": args.run_stacks,
            "same_test_splits": True,
        },
    }
    (args.report_dir / "comparison_with_market_models.json").write_text(json.dumps(out_json, indent=2), encoding="utf-8")

    md = [
        "# Comparison With Actual Market Models",
        "",
        markdown_table(rows),
        "",
        "## Winner By Metric",
    ]
    metrics = [
        "keypoint_f1_mean",
        "judge_detect_rate",
        "section_detect_rate",
        "decision_capture_rate",
        "rouge1",
        "rouge2",
        "rougeL",
    ]
    for m in metrics:
        best = max(rows, key=lambda x: getattr(x, m))
        md.append(f"- {m}: {best.name} ({getattr(best, m):.4f})")

    (args.report_dir / "comparison_with_market_models.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(out_json, indent=2))


if __name__ == "__main__":
    main()
