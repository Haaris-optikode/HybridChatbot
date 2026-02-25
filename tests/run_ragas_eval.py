"""
RAGAS Evaluation Framework for MedGraph AI RAG Pipeline.

Measures retrieval and generation quality using RAGAS metrics:
  - Faithfulness: Does the answer only use information from retrieved context?
  - Answer Relevancy: Is the answer relevant to the question?
  - Context Precision: Are the retrieved chunks relevant to the question?
  - Context Recall: Does the context contain the information needed to answer?

Usage:
    cd src
    python ../tests/run_ragas_eval.py                      # evaluate with default test set
    python ../tests/run_ragas_eval.py --output results.json # save results to file

Requires:
    pip install ragas datasets

The test dataset (tests/eval_dataset.json) contains sample Q&A pairs.
Edit it with real clinical Q&A to get meaningful results.
"""
import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

# ── Path setup ───────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ragas_eval")


def load_eval_dataset(path: str) -> list[dict]:
    """Load the evaluation dataset from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_retrieval(questions: list[str]) -> list[list[str]]:
    """Run each question through the RAG retrieval pipeline, return contexts."""
    from agent_graph.tool_clinical_notes_rag import get_rag_tool_instance

    rag = get_rag_tool_instance()
    all_contexts = []
    for q in questions:
        try:
            result = rag.search(q)
            # Split by the document separator used in search()
            chunks = [c.strip() for c in result.split("\n\n---\n\n") if c.strip()]
            all_contexts.append(chunks)
        except Exception as e:
            logger.warning("Retrieval failed for '%s': %s", q, e)
            all_contexts.append([])
    return all_contexts


def run_generation(questions: list[str], contexts: list[list[str]]) -> list[str]:
    """Generate answers using the RAG pipeline's LLM given contexts."""
    from langchain_openai import ChatOpenAI
    from agent_graph.load_tools_config import LoadToolsConfig

    cfg = LoadToolsConfig()
    model = cfg.clinical_notes_rag_summarization_llm
    llm = ChatOpenAI(model=model, temperature=0.0, max_tokens=2048, request_timeout=60)

    answers = []
    for q, ctx in zip(questions, contexts):
        if not ctx:
            answers.append("No relevant information found.")
            continue
        context_str = "\n\n".join(ctx)
        prompt = (
            "Answer the following clinical question based ONLY on the provided context. "
            "If the context does not contain the answer, say so.\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {q}\n"
            "Answer:"
        )
        try:
            answer = llm.invoke(prompt).content
            answers.append(answer)
        except Exception as e:
            logger.warning("Generation failed for '%s': %s", q, e)
            answers.append(f"Error: {e}")
    return answers


def evaluate_with_ragas(dataset: list[dict], contexts: list[list[str]], answers: list[str]):
    """Run RAGAS evaluation and return scores."""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )

    # Build RAGAS-compatible dataset
    eval_data = {
        "question": [d["question"] for d in dataset],
        "answer": answers,
        "contexts": contexts,
        "ground_truth": [d.get("ground_truth", "") for d in dataset],
    }

    ds = Dataset.from_dict(eval_data)

    logger.info("Running RAGAS evaluation on %d samples...", len(dataset))
    result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="RAGAS Evaluation for MedGraph AI")
    parser.add_argument("--dataset", default=str(ROOT / "tests" / "eval_dataset.json"),
                        help="Path to evaluation dataset JSON")
    parser.add_argument("--output", default=None, help="Save results to JSON file")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="Only test retrieval, skip LLM generation")
    args = parser.parse_args()

    # Load dataset
    logger.info("Loading dataset from %s", args.dataset)
    dataset = load_eval_dataset(args.dataset)
    logger.info("Loaded %d evaluation samples", len(dataset))

    questions = [d["question"] for d in dataset]

    # Step 1: Retrieval
    logger.info("Running retrieval...")
    contexts = run_retrieval(questions)

    retrieval_stats = {
        "total_questions": len(questions),
        "questions_with_context": sum(1 for c in contexts if c),
        "avg_chunks_per_question": sum(len(c) for c in contexts) / max(len(contexts), 1),
    }
    logger.info("Retrieval stats: %s", json.dumps(retrieval_stats, indent=2))

    if args.retrieval_only:
        results = {"retrieval_stats": retrieval_stats, "timestamp": datetime.now().isoformat()}
    else:
        # Step 2: Generation
        logger.info("Running generation...")
        answers = run_generation(questions, contexts)

        # Step 3: RAGAS evaluation
        try:
            ragas_result = evaluate_with_ragas(dataset, contexts, answers)
            results = {
                "ragas_scores": {k: float(v) for k, v in ragas_result.items()},
                "retrieval_stats": retrieval_stats,
                "timestamp": datetime.now().isoformat(),
                "num_samples": len(dataset),
            }
            logger.info("RAGAS Scores:")
            for metric, score in results["ragas_scores"].items():
                logger.info("  %s: %.4f", metric, score)
        except Exception as e:
            logger.error("RAGAS evaluation failed: %s", e)
            results = {
                "error": str(e),
                "retrieval_stats": retrieval_stats,
                "timestamp": datetime.now().isoformat(),
            }

    # Output
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        logger.info("Results saved to %s", out_path)
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
