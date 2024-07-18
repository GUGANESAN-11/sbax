import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Replicate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from transformers import pipeline
import os
from dotenv import load_dotenv
import tempfile

load_dotenv()

st.set_page_config(
    page_title="SBA INFO SOLUTION",
    page_icon="sba_info_solutions_logo.jpg",  # You can add your icon here
    layout="wide",  # Wide layout similar to your second code snippet
)

st.markdown('# :white[SBA INFO SOLUTION] ', unsafe_allow_html=True)
st.markdown('## :white[Search Engine] ', unsafe_allow_html=True)


def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything about"]

    if 'past' not in st.session_state:
        st.session_state['past'] = [""]


def conversation_chat(query, chain, history):
    result = chain({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]


def display_chat_history():
    reply_container = st.container()
    container = st.container()

    # Ensure unique key for the form
    form_key = 'my_form_input_' + str(len(st.session_state['past']))

    with container:
        with st.form(key=form_key, clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about your Documents", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            with st.spinner('Generating response...'):
                output = conversation_chat(user_input, st.session_state['chain'], st.session_state['history'])

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")


def create_conversational_chain(vector_store):
    load_dotenv()
    llm = Replicate(
        streaming=True,
        model="replicate/llama-2-70b-chat:58d078176e02c219e11eb4da5a02a7830a283b14cf8f94537af893ccff5ee781",
        callbacks=[StreamingStdOutCallbackHandler()],
        input={"temperature": 0.01, "max_length": 500, "top_p": 1}
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                                  retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                                  memory=memory)
    return chain


def perform_summarization(text_chunks):
    texts = [chunk.page_content for chunk in text_chunks]
    full_text = " ".join(texts)
    summary = full_text[:1000]  # Dummy summarization for demonstration
    return summary


def perform_entity_extraction(text_chunks):
    ner = pipeline("ner", grouped_entities=True)
    texts = [chunk.page_content for chunk in text_chunks]
    full_text = " ".join(texts)
    entities = ner(full_text)
    entity_list = [entity['word'] for entity in entities]
    return ", ".join(set(entity_list))


def main():
    load_dotenv()
    initialize_session_state()
    st.sidebar.image("sba_info_solutions_logo.jpg", width=200, use_column_width=False)
    st.sidebar.markdown("---")
    st.sidebar.title("Document Processing")
    uploaded_files = st.sidebar.file_uploader("Upload files", accept_multiple_files=True)

    if uploaded_files:
        text = []
        for file in uploaded_files:
            file_extension = os.path.splitext(file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name

            loader = None
            if file_extension == ".pdf":
                loader = PyPDFLoader(temp_file_path)
            elif file_extension == ".docx" or file_extension == ".doc":
                loader = Docx2txtLoader(temp_file_path)
            elif file_extension == ".txt":
                loader = TextLoader(temp_file_path)

            if loader:
                text.extend(loader.load())
                os.remove(temp_file_path)

        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=100, length_function=len)
        text_chunks = text_splitter.split_documents(text)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                           model_kwargs={'device': 'cpu'})

        vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

        st.session_state['chain'] = create_conversational_chain(vector_store)

        st.sidebar.header("Tasks")
        if st.sidebar.button("Summarization"):
            with st.spinner("Summarizing..."):
                result = perform_summarization(text_chunks)
            st.session_state['past'].append("Summarization")
            st.session_state['generated'].append(result)
            display_chat_history()

        # if st.sidebar.button("Entity Extraction"):
        #     with st.spinner("Extracting Entities..."):
        #         result = perform_entity_extraction(text_chunks)
        #     st.session_state['past'].append("Entity Extraction")
        #     st.session_state['generated'].append(result)
        #     display_chat_history()

        display_chat_history()


if __name__ == "__main__":
    main()
