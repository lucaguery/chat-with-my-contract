import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from streamlit_chat import message

from ingest_data import vectorstore


@st.cache_resource
def connect_to_vectorstore():
    return vectorstore


@st.cache_resource
def load_chain(_vectorstore):
    rag_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature=0.0, model_name="gpt-3.5-turbo"),
        retriever=_vectorstore.as_retriever(),
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
    result = rag_chain(
        {"question": user_input, "chat_history": st.session_state["generated"]}
    )
    response = result["answer"]
    st.session_state["past"].append(user_input)
    st.session_state["generated"].append((user_input, result["answer"]))

if st.session_state.get("generated"):
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
        message(st.session_state["generated"][i][1], key=str(i))
