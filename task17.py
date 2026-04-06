from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_react_agent, AgentExecutor

def rag_agent(question: str) -> str:

    docs = [Document(page_content=d) for d in RAG_DOCUMENTS]
    embeddings = OpenAIEmbeddings()

    vectorstore = PGVector.from_documents(
        docs,
        embedding=embeddings,
        collection_name="rag_collection",
        connection_string="postgresql+psycopg2:///postgres:Password%40123@localhost:5432/db"
    )

    retriever = vectorstore.as_retriever()

    tool = create_retriever_tool(
        retriever,
        name="knowledge_base",
        description="Search technical knowledge"
    )

    llm = ChatOpenAI()

    agent = create_react_agent(llm, [tool])
    executor = AgentExecutor(agent=agent, tools=[tool])

    result = executor.invoke({"input": question})

    return result["output"]