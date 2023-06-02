import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
import streamlit as st
from streamlit_chat import message
from utils import get_initial_message, update_chat
import json
import uuid
import base64
st.set_page_config(layout="wide",page_title="Pet Care QA Chatbot", page_icon="🐾")
st.markdown(
    "<h1 style='text-align: center; color: red;'>UK and Ireland DoS AI Test Finder BETA</h1>",
    unsafe_allow_html=True
)

@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img = get_img_as_base64("images_half.png")
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
    background-image: url("data:image/png;base64,{img}");
    background-size: cover;
    background-position: top left;
    background-repeat: no-repeat;
    background-attachment: local;
}}

[data-testid="stHeader"] {{
    background: rgba(0, 0, 0, 0);
}}

.sidebar .stMarkdown {{
    height: calc(100vh - 8rem);
    overflow-y: auto;
}}
[data-testid="stFormSubmitButton"]{{
     width: 10%;
    bottom: 3rem;
    position: fixed;
    z-index: 1;
    left:80%;
}}
[data-testid="stForm"]
{{
     width: 100%;
    height:1%;
    border:0px;
}}

</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Define the sidebar content
st.sidebar.title('something here')
st.sidebar.markdown('''
    ## About
    you can write here a description
    here 
''')
if st.sidebar.button("Clear Chat"):
    st.session_state["messages"] = get_initial_message()
    st.session_state["past"] = []
    st.session_state["generated"] = []
# Initialize the QA chain
os.environ["OPENAI_API_KEY"] = "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"
persist_directory = 'db'

def init():
    embedding = OpenAIEmbeddings()
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    retriever = vectordb.as_retriever()

    qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(temperature=0.2, streaming=True),
                                           chain_type="refine",
                                           retriever=retriever,
                                           return_source_documents=True)
    return qa_chain

qa_chain = init()

# Define the chat functionality
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []

if "messages" not in st.session_state:
    st.session_state["messages"] = get_initial_message()

# Styling for chat input and submit button
styl = """
<style>
.stTextInput {
    width: 50%;
    position: fixed;
    bottom: 3rem;
    z-index: 1;
    left: 30%;
}


.stButton > button:nth-child(2) {
    right: 100;
}  
</style>
"""
st.markdown(styl, unsafe_allow_html=True)

# Chat input and submit button
container = st.container()
response_container = st.container()
with container:
    with st.form(key='my_form', clear_on_submit=True):
        query_text = st.text_input("Type Your Question:", key="input")
        submit_button = st.form_submit_button(label='Send')

# Process user input and display chat messages
if query_text and submit_button:
    with st.spinner("Generating..."):
        messages = st.session_state["messages"]
        messages = update_chat(messages, "user", query_text)
        llm_response = qa_chain(query_text)
        response = llm_response['result']
        st.session_state.past.append(query_text)  # Append the new user message at the end
        st.session_state.generated.append(response)  # Append the new generated message at the end

        # Display the chat messages with newest messages at the top
        if len(st.session_state["generated"]) > 0:
            with response_container:
                for i in range(len(st.session_state["generated"]) - 1, -1, -1):
                    index = len(st.session_state["generated"]) - i - 1  # Calculate the index for displaying messages
                    message(st.session_state["past"][index], is_user=True, key=str(uuid.uuid4()) + "_user")
                    message(st.session_state["generated"][index], key=str(uuid.uuid4()))
