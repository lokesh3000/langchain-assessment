from langsmith.evaluation import evaluate

def run_langsmith_evaluation() -> dict:

    def target(inputs: dict) -> dict:
        return {
            "answer": basic_rag_pipeline(RAG_DOCUMENTS, inputs["question"])
        }

    def evaluator(run, example):
        pred = run.outputs["answer"].lower()
        expected = example.outputs["answer"].lower()
        return {"score": expected in pred}

    results = evaluate(
        target,
        data="rag-eval-dataset",
        evaluators=[evaluator],
        experiment_prefix="rag-eval"
    )

    return {
        "dataset": "rag-eval-dataset",
        "num_examples": len(results),
        "pass_rate": sum(r["score"] for r in results) / len(results)
    }