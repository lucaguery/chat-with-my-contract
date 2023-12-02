import pickle

import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from streamlit_chat import message

from ingest_data import vectorstore

# with open("vectorstore.pkl", "rb") as f:
#     vectorstore = pickle.load(f)


@st.cache_resource
def connect_to_vectorstore():
    return vectorstore


@st.cache_resource
def load_chain(_vectorstore):
    from langchain.prompts import ChatPromptTemplate
    from langchain.chains import (
    StuffDocumentsChain, LLMChain, ConversationalRetrievalChain
)
    template = """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise.
    Question: {question} 
    Context: {context} 
    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)
    retriever = _vectorstore.as_retriever()
    print(retriever)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # rag_chain = (
    #     {"context": retriever, "question": RunnablePassthrough()}
    #     | prompt
    #     | llm
    #     | StrOutputParser()
    # )

    rag_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature=0.0, model_name="gpt-3.5-turbo"),
        retriever=vectorstore.as_retriever(),
    )
    return rag_chain


st.title("Chat With Your Contract!")

vector_store = connect_to_vectorstore()

rag_chain = load_chain(vector_store)

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


user_input = st.text_input("You: ", key="input")

if user_input:
    # result = rag_chain.invoke(user_input)
    result = rag_chain({"question": user_input, "chat_history": st.session_state["generated"]})
    response = result['answer']
    st.session_state["past"].append(user_input)
    st.session_state["generated"].append((user_input, result['answer']))

if st.session_state.get("generated"):
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
        message(st.session_state["generated"][i][1], key=str(i))
