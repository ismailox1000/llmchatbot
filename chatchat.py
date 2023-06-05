import os
import streamlit as st
import base64
import json
import uuid

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from streamlit_chat import message
from utils import get_initial_message, update_chat

# Set Streamlit page configuration
st.set_page_config(layout="wide", page_title="Pet Care QA Chatbot", page_icon="🐾")

# Set background image for the app
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
    background-attachment: fixed; /* Changed from local to fixed */
}}

[data-testid="stHeader"] {{
    background: rgba(0, 0, 0, 0);
}}

.sidebar .stMarkdown {{
    height: calc(100vh - 8rem);
    overflow-y: auto;
}}

[data-testid="stFormSubmitButton"] {{
    width: 10%;
    bottom: 3rem;
    position: fixed;
    z-index: 1;
    left: 80%;
}}

[data-testid="stForm"] {{
    width: 100%;
    height: 1%;
    border: 0px;
}}

[data-testid="main-menu-popover"]{{

    background-color: white;
}}
/* Media Query for Mobile */
@media only screen and (max-width: 800px) {{
    [data-testid="stAppViewContainer"] > .main {{
        background-position: center; /* Adjusted to show the right half of the picture */
        background-size: 280%
    }}
}}

</style>


"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Define the sidebar content




st.sidebar.title('About')
st.sidebar.markdown('''Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vestibulum enim mi, vestibulum in dapibus vitae, commodo non velit. Nulla maximus purus nunc, nec suscipit neque condimentum at. In dapibus, felis at tristique suscipit, sapien urna pretium felis, id interdum ex.''')

st.sidebar.markdown('''------------------------''')

if st.sidebar.button("Clear Chat", help="clear your history"):
    st.session_state["messages"] = get_initial_message()
    st.session_state["past"] = []
    st.session_state["generated"] = []

st.sidebar.markdown('''------------------------''')


st.sidebar.markdown('''
    For further support and guidance follow the Link bellow
    
     
''')

columns = st.sidebar.columns(6)

with columns[0]:
    st.write("""<div style="width:100%;"><a href="https://www.idexx.co.uk/en-gb/veterinary/reference-laboratories/reference-laboratory-support/">https://www.idexx.co.uk/en-gb/veterinary...</a></div>""", unsafe_allow_html=True)

# Initialize the QA chain
os.environ["OPENAI_API_KEY"] = "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"
persist_directory = 'db'

def init():
    embedding = OpenAIEmbeddings()
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    retriever = vectordb.as_retriever()

    qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(temperature=0.2,max_tokens=512, streaming=True),
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

        pages = []
        for num in llm_response['source_documents']:
            page_num = num.metadata['page']
            pages.append(str(page_num))

        source_pages = "source pages : "+ ", ".join(set(pages))
        st.session_state.past.append(query_text)  # Append the new user message at the end
        st.session_state.generated.append(response + '\n' + source_pages)  # Append the new generated message at the end

        # Display the chat messages with newest messages at the top
        if len(st.session_state["generated"]) > 0:
            with response_container:
                for i in range(len(st.session_state["generated"]) - 1, -1, -1):
                    index = len(st.session_state["generated"]) - i - 1  # Calculate the index for displaying messages
                    
                    message(st.session_state["past"][index], is_user=True, key=str(uuid.uuid4()) + "_user",avatar_style="micah")
                    message(st.session_state["generated"][index], key=str(uuid.uuid4()),avatar_style="personas")
