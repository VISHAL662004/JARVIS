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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.services.entity_extractor import EntityExtractionService
from app.services.keypoint_extractor import KeyPointExtractionService
from app.services.summarizer import SummarizationService
from app.utils.text import remove_boilerplate, sentence_split


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def lead3_summary(text: str, max_sentences: int = 3) -> str:
    sents = sentence_split(remove_boilerplate(text))
    return " ".join(sents[:max_sentences])


def textrank_style_summary(text: str, max_sentences: int = 4) -> str:
    sents = sentence_split(remove_boilerplate(text))
    if not sents:
        return ""
    if len(sents) <= max_sentences:
        return " ".join(sents)

    vec = TfidfVectorizer(max_features=12000, ngram_range=(1, 2))
    mat = vec.fit_transform(sents)
    doc_vec = mat.mean(axis=0)
    doc_vec = getattr(doc_vec, "A", doc_vec)
    sim = cosine_similarity(mat, doc_vec).reshape(-1)
    top_ids = np.argsort(sim)[-max_sentences:]
    top_ids = sorted(int(i) for i in top_ids)
    return " ".join(sents[i] for i in top_ids)


KEYWORDS = {
    "FACT": ["fact", "background", "history", "brief facts"],
    "ISSUE": ["issue", "question", "point for determination"],
    "ARGUMENT": ["submitted", "contended", "argued", "counsel"],
    "REASONING": ["because", "therefore", "held", "in view of", "reason"],
    "DECISION": ["allowed", "dismissed", "ordered", "directed", "granted", "disposed"],
}


def keyword_keypoints(text: str) -> set[str]:
    labels = set()
    for sent in sentence_split(remove_boilerplate(text)):
        low = sent.lower()
        for label, words in KEYWORDS.items():
            if any(w in low for w in words):
                labels.add(label)
    return labels


JUDGE_PAT = re.compile(r"\b(?:JUSTICE|J\.)\s+[A-Z][A-Za-z.\s]{2,}", re.IGNORECASE)
SECTION_PAT = re.compile(r"\b(?:Section|Sec\.|Article)\s+[0-9A-Za-z()/-]+", re.IGNORECASE)
FINAL_PAT = re.compile(r"\b(allowed|dismissed|ordered|directed|granted|modified|released)\b", re.IGNORECASE)


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


def eval_pipeline_model(key_test: list[dict], sum_test: list[dict]) -> EvalRow:
    ent = EntityExtractionService()
    key = KeyPointExtractionService()
    summ = SummarizationService()

    grouped: dict[str, list[dict]] = {}
    for row in key_test:
        grouped.setdefault(row["doc_id"], []).append(row)

    key_f1, judge, section, decision = [], [], [], []
    for _, rows in tqdm(list(grouped.items())[:400], desc="Pipeline eval", unit="doc"):
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
    for row in tqdm(sum_test[:300], desc="Pipeline summary", unit="doc"):
        pred = summ.summarize_abstractive(remove_boilerplate(row["source"]))
        rouge_vals.append(scorer.score(row["target"], pred))

    return EvalRow(
        name="Your Pipeline (Scratch + Heuristic Fallback)",
        keypoint_f1_mean=float(np.mean(key_f1)) if key_f1 else 0.0,
        judge_detect_rate=float(np.mean(judge)) if judge else 0.0,
        section_detect_rate=float(np.mean(section)) if section else 0.0,
        decision_capture_rate=float(np.mean(decision)) if decision else 0.0,
        rouge1=float(np.mean([x["rouge1"].fmeasure for x in rouge_vals])) if rouge_vals else 0.0,
        rouge2=float(np.mean([x["rouge2"].fmeasure for x in rouge_vals])) if rouge_vals else 0.0,
        rougeL=float(np.mean([x["rougeL"].fmeasure for x in rouge_vals])) if rouge_vals else 0.0,
    )


def eval_rule_baseline(key_test: list[dict], sum_test: list[dict]) -> EvalRow:
    grouped: dict[str, list[dict]] = {}
    for row in key_test:
        grouped.setdefault(row["doc_id"], []).append(row)

    key_f1, judge, section, decision = [], [], [], []
    for _, rows in tqdm(list(grouped.items())[:400], desc="Rule baseline", unit="doc"):
        text = " ".join(r["sentence"] for r in rows)
        gold_labels = {r["label"] for r in rows}
        pred_labels = keyword_keypoints(text)
        key_f1.append(doc_level_f1(gold_labels, pred_labels))

        judge.append(float(bool(JUDGE_PAT.search(text))))
        section.append(float(bool(SECTION_PAT.search(text))))
        decision.append(float(bool(FINAL_PAT.search(text))))

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_vals = []
    for row in tqdm(sum_test[:300], desc="Rule summary", unit="doc"):
        pred = textrank_style_summary(row["source"], max_sentences=4)
        rouge_vals.append(scorer.score(row["target"], pred))

    return EvalRow(
        name="Existing Rule-Based + TextRank-Style Baseline",
        keypoint_f1_mean=float(np.mean(key_f1)) if key_f1 else 0.0,
        judge_detect_rate=float(np.mean(judge)) if judge else 0.0,
        section_detect_rate=float(np.mean(section)) if section else 0.0,
        decision_capture_rate=float(np.mean(decision)) if decision else 0.0,
        rouge1=float(np.mean([x["rouge1"].fmeasure for x in rouge_vals])) if rouge_vals else 0.0,
        rouge2=float(np.mean([x["rouge2"].fmeasure for x in rouge_vals])) if rouge_vals else 0.0,
        rougeL=float(np.mean([x["rougeL"].fmeasure for x in rouge_vals])) if rouge_vals else 0.0,
    )


def eval_lead3_baseline(key_test: list[dict], sum_test: list[dict]) -> EvalRow:
    grouped: dict[str, list[dict]] = {}
    for row in key_test:
        grouped.setdefault(row["doc_id"], []).append(row)

    key_f1, judge, section, decision = [], [], [], []
    for _, rows in tqdm(list(grouped.items())[:400], desc="Lead-3 baseline", unit="doc"):
        text = " ".join(r["sentence"] for r in rows)
        gold_labels = {r["label"] for r in rows}

        # Lead baseline assumes early document sentences represent factual + issue context.
        sents = sentence_split(remove_boilerplate(text))[:3]
        low = " ".join(sents).lower()
        pred_labels = set()
        if sents:
            pred_labels.add("FACT")
        if "issue" in low or "question" in low:
            pred_labels.add("ISSUE")

        key_f1.append(doc_level_f1(gold_labels, pred_labels))
        judge.append(float(bool(JUDGE_PAT.search(text))))
        section.append(float(bool(SECTION_PAT.search(text))))
        decision.append(float(bool(FINAL_PAT.search(text[-1500:]))))

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_vals = []
    for row in tqdm(sum_test[:300], desc="Lead-3 summary", unit="doc"):
        pred = lead3_summary(row["source"], max_sentences=3)
        rouge_vals.append(scorer.score(row["target"], pred))

    return EvalRow(
        name="Existing Lead-3 Extractive Baseline",
        keypoint_f1_mean=float(np.mean(key_f1)) if key_f1 else 0.0,
        judge_detect_rate=float(np.mean(judge)) if judge else 0.0,
        section_detect_rate=float(np.mean(section)) if section else 0.0,
        decision_capture_rate=float(np.mean(decision)) if decision else 0.0,
        rouge1=float(np.mean([x["rouge1"].fmeasure for x in rouge_vals])) if rouge_vals else 0.0,
        rouge2=float(np.mean([x["rouge2"].fmeasure for x in rouge_vals])) if rouge_vals else 0.0,
        rougeL=float(np.mean([x["rougeL"].fmeasure for x in rouge_vals])) if rouge_vals else 0.0,
    )


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare current model with existing legal analysis baselines")
    parser.add_argument("--split-dir", type=Path, default=Path("data/processed/splits"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    key_test = load_jsonl(args.split_dir / "keypoints_test.jsonl")
    sum_test = load_jsonl(args.split_dir / "summary_test.jsonl")

    rows = [
        eval_pipeline_model(key_test, sum_test),
        eval_rule_baseline(key_test, sum_test),
        eval_lead3_baseline(key_test, sum_test),
    ]

    args.report_dir.mkdir(parents=True, exist_ok=True)
    out_json = {
        "rows": [r.as_dict() for r in rows],
        "notes": [
            "All models are evaluated on identical test splits.",
            "This comparison is fully reproducible from local artifacts in this repository.",
            "Rule/TextRank and Lead-3 are standard existing baseline families for legal NLP pipelines.",
        ],
    }
    (args.report_dir / "comparison_with_existing_models.json").write_text(json.dumps(out_json, indent=2), encoding="utf-8")

    md = [
        "# Comparison With Existing Legal Analysis Baselines",
        "",
        markdown_table(rows),
        "",
        "## Winner By Metric",
    ]

    metric_names = [
        "keypoint_f1_mean",
        "judge_detect_rate",
        "section_detect_rate",
        "decision_capture_rate",
        "rouge1",
        "rouge2",
        "rougeL",
    ]
    for m in metric_names:
        best = max(rows, key=lambda x: getattr(x, m))
        md.append(f"- {m}: {best.name} ({getattr(best, m):.4f})")

    (args.report_dir / "comparison_with_existing_models.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(out_json, indent=2))


if __name__ == "__main__":
    main()
