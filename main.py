import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from io import StringIO
from langchain.document_loaders import TextLoader

def remove_dates(text):
    text = text.split('-')
    text = text[1:]
    text = ' '.join(text)
    return text

def get_pdf_text(pdf_docs):
    text=''
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=5000,
        chunk_overlap=1000,
        length_function=len,
    )
    
    if type(text) == str:
        text_chunks = splitter.split_text(text)
        return text_chunks
    else:
        try:
            loader = TextLoader(text)
            loader = loader.load()
            st.write(loader)
            text_chunks = splitter.split_text(loader)
            return text_chunks
        except:
            stringio = StringIO(text[0].getvalue().decode("utf-8"))
            string_data = stringio.read()
            text=" ".join(string_data.split())          
            text=remove_dates(text)  
            st.write(len(text)) 
            text_chunks = splitter.split_text(text)
            return text_chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(client=None)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(client=None)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)
    return conversation_chain

def handle_userinput(user_question):
    if 'conversation' not in st.session_state:
        st.error("Please process your documents first",icon="🚨")
    else:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']


            


def main():
    load_dotenv()


    st.set_page_config(
        page_title="PDF GPT",
        page_icon="🔗",
    )

    st.title("PDF GPT")

    st.header("Chat with multiple documents :books:")


    user_question = st.chat_input("Ask a question about your documents")
    if user_question:
        handle_userinput(user_question)
        
        for i in range(len(st.session_state.chat_history)):
            if i%2 == 0:
                with st.chat_message("user"):
                    st.write(st.session_state.chat_history[i].content)
            else:
                with st.chat_message("assistant"):
                # check if its the last message 
                    if i == len(st.session_state.chat_history)-1:
                        msg = st.empty()
                        full_msg = ""
                        for j in range(len(st.session_state.chat_history[i].content)):
                            full_msg += st.session_state.chat_history[i].content[j]
                            msg.write(full_msg)
                         
                    else:
                        st.write(st.session_state.chat_history[i].content)

    
    with st.sidebar:
        st.subheader("Your documents")

        # option to select between pdf and text file 
        doc_type = st.radio("Upload your documents in:", ("PDF", "Text"), index=0)

        if doc_type:
            docs = st.file_uploader(
            "Upload your documents here and click on 'Process'", accept_multiple_files=True)
            
        if st.button("Process"):
            with st.spinner("Processing"):
                if doc_type == "PDF":
                    # get pdf text
                    raw_text = get_pdf_text(docs)

                    # get the text chunks
                    text_chunks = get_text_chunks(raw_text)

                    # create vector store
                    vectorstore = get_vectorstore(text_chunks)

                    # create conversation chain
                    st.session_state.conversation = get_conversation_chain(vectorstore)

                if doc_type == "Text":
                    # get text chunks
                    text_chunks = get_text_chunks(docs)

                    # create vector store
                    vectorstore = get_vectorstore(text_chunks)

                    # create conversation chain
                    st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == '__main__':
    main()
        