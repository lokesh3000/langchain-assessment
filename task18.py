import os
from langchain_core.tracers.context import collect_runs

def traced_chain(topic: str) -> dict:

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "rag-project"

    prompt = ChatPromptTemplate.from_template("Explain {topic}")
    llm = ChatOpenAI()

    chain = prompt | llm | StrOutputParser()

    with collect_runs() as cb:
        result = chain.invoke(
            {"topic": topic},
            config={"run_name": "task18_trace", "tags": ["challenge"]}
        )

    run_id = str(cb.traced_runs[0].id)

    return {"answer": result, "run_id": run_id}