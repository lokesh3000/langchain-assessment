from langsmith import Client

def create_langsmith_dataset() -> str:

    client = Client()

    dataset = client.create_dataset("rag-eval-dataset")

    questions = [
        "What does RAG stand for?",
        "What PostgreSQL extension enables vector search?",
        "What LangChain tool provides observability?"
    ]

    answers = [
        "Retrieval-Augmented Generation",
        "pgvector",
        "LangSmith"
    ]

    client.create_examples(
        inputs=[{"question": q} for q in questions],
        outputs=[{"answer": a} for a in answers],
        dataset_id=dataset.id
    )

    return str(dataset.id)