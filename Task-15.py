from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_postgres import PGVector
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def rag_with_sources(documents: list, question: str) -> dict:
    docs = [Document(page_content=d) for d in documents]

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = PGVector.from_documents(
        docs,
        embedding=embeddings,
        collection_name="rag_collection_sources",
        connection="postgresql+psycopg://postgres:postgres@localhost:5432/postgres"
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_template(
        "Answer using only this context:\n{context}\n\nQuestion: {question}"
    )

    llm = ChatOpenAI(model="gpt-4o-mini")

    retrieval = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    )

    def generate(inputs):
        docs = inputs["context"]
        context = format_docs(docs)
        answer = (
            prompt
            | llm
            | StrOutputParser()
        ).invoke({"context": context, "question": inputs["question"]})
        sources = [{"content": d.page_content, "score": 1.0} for d in docs]
        return {"answer": answer, "sources": sources}

    chain = retrieval | generate

    return chain.invoke(question)