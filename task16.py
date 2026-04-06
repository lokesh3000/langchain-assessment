from langchain.chains import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage

def conversational_rag(documents: list) -> list:

    docs = [Document(page_content=d) for d in documents]
    embeddings = OpenAIEmbeddings()

    vectorstore = PGVector.from_documents(
        docs,
        embedding=embeddings,
        collection_name="rag_collection",
        connection_string="postgresql+psycopg2://postgres:Password%40123@localhost:5432/db"
    )

    retriever = vectorstore.as_retriever()

    llm = ChatOpenAI()

    contextualize_prompt = ChatPromptTemplate.from_template(
        "Given chat history and a follow-up question, rewrite it as standalone.\n"
        "Chat history: {chat_history}\nQuestion: {input}"
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )

    qa_prompt = ChatPromptTemplate.from_template(
        "Answer based on context:\n{context}\n\nQuestion: {input}"
    )

    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    chat_history = []

    # Turn 1
    res1 = rag_chain.invoke({"input": "What is LangChain?", "chat_history": chat_history})
    chat_history.append(HumanMessage(content="What is LangChain?"))
    chat_history.append(AIMessage(content=res1["answer"]))

    # Turn 2
    res2 = rag_chain.invoke({"input": "What version introduced LCEL?", "chat_history": chat_history})

    return [res1["answer"], res2["answer"]]