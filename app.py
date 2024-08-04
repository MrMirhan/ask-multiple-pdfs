import os
import re
import requests
import streamlit as st
from uuid import uuid4
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from googlesearch import search
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import AIMessage, HumanMessage
from htmlTemplates import css, bot_template, user_template
from streamlit_session_browser_storage import SessionStorage
import faiss

# Load environment variables
load_dotenv()

# Validate environment variables
using_ai = os.getenv("USE")
if not using_ai:
    st.error("Please set the USE environment variable in the .env file.")
    st.stop()

required_vars = {
    "openai": ["OPENAI_API_KEY", "OPENAI_MODEL_NAME", "OPENAI_EMBEDDINGS_MODEL_NAME"],
    "huggingface": ["HUGGINGFACEHUB_API_KEY", "HUGGINGFACE_MODEL_NAME", "HUGGINGFACE_EMBEDDINGS_MODEL_NAME"]
}

if using_ai not in required_vars:
    st.error("Invalid USE value. It should be 'openai' or 'huggingface'.")
    st.stop()

missing_vars = [var for var in required_vars[using_ai] if not os.getenv(var)]
if missing_vars:
    st.error(f"Please set the following environment variables in the .env file: {', '.join(missing_vars)}")
    st.stop()

import openai

# Set API keys
if using_ai == "openai":
    openai.api_key = os.getenv("OPENAI_API_KEY")

openai_model_name = os.getenv("OPENAI_MODEL_NAME")
huggingface_model_name = os.getenv("HUGGINGFACE_MODEL_NAME")
openai_embeddings_model_name = os.getenv("OPENAI_EMBEDDINGS_MODEL_NAME")
huggingface_embeddings_model_name = os.getenv("HUGGINGFACE_EMBEDDINGS_MODEL_NAME")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    return text_splitter.split_text(text)

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(model=openai_embeddings_model_name) if using_ai == "openai" else HuggingFaceInstructEmbeddings(model_name=huggingface_embeddings_model_name)
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(model=openai_model_name) if using_ai == "openai" else HuggingFaceHub(repo_id=huggingface_model_name, model_kwargs={"temperature": 0.5, "max_length": 4000})
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)

def search_google(query, num_results=5):
    return list(search(query, num=num_results, stop=num_results, lang="en"))

def extract_link_content(links):
    contents = []
    for link in links:
        try:
            response = requests.get(link)
            soup = BeautifulSoup(response.text, 'html.parser')
            content = " ".join(soup.get_text()[:30000].split()).replace("\n", " ").replace("\r", " ").replace("\t", " ").replace("\'", "").replace("\"", "")
            content = re.sub('<.*?>', '', content)[:6000]
            meta_title = soup.find('meta', property='og:title')
            meta_description = soup.find('meta', property='og:description')
            contents.append({
                "content": content,
                "link": link,
                "meta_title": meta_title['content'] if meta_title else "None",
                "meta_description": meta_description['content'] if meta_description else "None"
            })
        except Exception as e:
            print(f"Error extracting summary for {link}: {str(e)}")
    return contents

def summarize_content(content, question, query, meta_title, meta_desc, link):
    try:
        response = requests.get(link)
        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.find('title').text if soup.find('title') else "No title found"
        content = " ".join(soup.get_text()[:30000].split()).replace("\n", " ").replace("\r", " ").replace("\'", "").replace("\"", "")
        content = re.sub('<.*?>', '', content)[:6000]
        return f"[{title}]({link}):\n{content}"
    except Exception as e:
        print(f"Error summarizing content from {link}: {str(e)}")
        return "Error summarizing content"

def give_most_relevant_answer(content, question):
    prompt = f"User Question: {question}\n\n{content}\n\nPlease provide the best answer to the user's question, considering the information given and be helpful to user.\n\nAnswer:"
    try:
        if using_ai == "openai":
            response = openai.Completion.create(
                model=openai_model_name,
                prompt=prompt,
                max_tokens=150,  # Adjust max_tokens based on your needs
                temperature=0.7
            )
            answer = response.choices[0].text.strip()
        else:
            response = HuggingFaceHub(repo_id=huggingface_model_name, model_kwargs={"temperature": 0.5, "max_length": 4000}).generate([prompt])
            answer = response.generations[0].text.strip()
    except Exception as e:
        print(e)
        answer = ""
    return answer

def question_should_be_search_on_google(question, pdf_response):
    prompt = f"User Question:\n{question}\n\nAnswer from VectorDB based on uploaded PDF files from user:\n{pdf_response}\n\nShould this question be searched on Google for better response? If it can be searched on Google, respond with 'Yes'; otherwise, respond with 'No'. Only respond with 'Yes' or 'No'."
    try:
        if using_ai == "openai":
            response = openai.Completion.create(
                model=openai_model_name,
                prompt=prompt,
                max_tokens=10,  # Adjust max_tokens based on your needs
                temperature=0.0
            )
            answer = response.choices[0].text.strip()
        else:
            response = HuggingFaceHub(repo_id=huggingface_model_name, model_kwargs={"temperature": 0.5, "max_length": 4000}).generate([prompt])
            answer = response.generations[0].text.strip()
    except Exception as e:
        print(e)
        answer = "No"
    return answer.lower() == "yes"

def what_should_be_the_query_for_google(question, pdf_response):
    prompt = f"User Question:\n{question}\n\nAnswer from VectorDB based on uploaded PDF files from user:\n{pdf_response}\n\nWhat should be the search query for Google to get the best response? Please provide only the search query that will be used to search on Google. Don't provide any other information."
    try:
        if using_ai == "openai":
            response = openai.Completion.create(
                model=openai_model_name,
                prompt=prompt,
                max_tokens=50,  # Adjust max_tokens based on your needs
                temperature=0.7
            )
            answer = response.choices[0].text.strip()
        else:
            response = HuggingFaceHub(repo_id=huggingface_model_name, model_kwargs={"temperature": 0.5, "max_length": 4000}).generate([prompt])
            answer = response.generations[0].text.strip()
    except Exception as e:
        print(e)
        answer = question
    return answer

def message_to_dict(message):
    return {
        "type": type(message).__name__,
        "content": message.content
    }

def dict_to_message(message_dict):
    if not isinstance(message_dict, dict):  # Ensure the input is a dictionary
        raise TypeError("Expected a dictionary for message_dict")
    
    if "type" not in message_dict or "content" not in message_dict:
        raise ValueError("Dictionary must contain 'type' and 'content' keys")
    
    if message_dict["type"] == "HumanMessage":
        return HumanMessage(content=message_dict["content"])
    elif message_dict["type"] == "AIMessage":
        return AIMessage(content=message_dict["content"])
    else:
        raise ValueError(f"Unknown message type: {message_dict['type']}")

def handle_userinput(user_question):
    if not user_question:
        st.warning("Please ask a question.")
        return

    # Get the response from the conversation chain
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    # Convert chat history to a serializable format
    serialized_chat_history = [message_to_dict(message) for message in st.session_state.chat_history]
    st.session_state.chat_history = serialized_chat_history

    # Debugging: Print serialized chat history
    #st.write(f"Debug: Serialized chat history = {st.session_state.chat_history}")

    # Filter out any empty messages
    filtered_chat_history = [
        msg_dict for msg_dict in st.session_state.chat_history 
        if isinstance(msg_dict, dict) and msg_dict.get('content')
    ]

    # Convert the filtered chat history back to message objects
    try:
        st.session_state.chat_history = [dict_to_message(msg_dict) for msg_dict in filtered_chat_history[::-1]]
    except Exception as e:
        st.error(f"Error converting dictionary to message: {e}")
        return

    # Check the last message and its content
    if not st.session_state.chat_history:
        st.error("No messages found in chat history.")
        return

    last_message = st.session_state.chat_history[-1]
    if not hasattr(last_message, 'content'):
        st.error(f"Last message does not have 'content' attribute: {last_message}")
        return

    combined_content = f"Current PDF Response for question:\n{last_message.content}"

    if len(st.session_state.chat_history) > 2:
        combined_content += "\n\nPrevious Conversation with the user:\n"
        combined_content += "User Question: " + st.session_state.chat_history[-4].content + "\n"
        combined_content += "Bot Answer: " + st.session_state.chat_history[-3].content + "\n"

    should_search = question_should_be_search_on_google(user_question, last_message.content)

    if should_search:
        google_query = what_should_be_the_query_for_google(user_question, last_message.content)
        links = search_google(google_query)
        contents = extract_link_content(links)

        summaries = [summarize_content(content["content"], user_question, google_query, content["meta_title"], content["meta_description"], content["link"]) for content in contents]

        combined_content += "\n\nUser question searched on Google for better results."
        combined_content += "\n\nGoogle Search Query: " + google_query
        combined_content += "\n\nHere are some relevant search summaries from the Google searches:\n"
        combined_content += "\n\n".join([f"Summary {i+1}:\n{summary}" for i, summary in enumerate(summaries)])
        google_search_status = "Searched on Google"
    else:
        combined_content += "\n\nUser question was not searched on Google."
        google_search_status = "Not Searched on Google"

    most_relevant_answer = give_most_relevant_answer(combined_content, user_question)

    # Create a new AI message
    ai_msg = AIMessage(content=most_relevant_answer)
    message_id = str(uuid4())

    # Append new AI message to chat history
    st.session_state.chat_history.append(ai_msg)

    # Debugging: Print updated chat history
    # st.write(f"Debug: Updated chat history = {[message_to_dict(msg) for msg in st.session_state.chat_history]}")

    # Display updated chat history
    for message in st.session_state.chat_history:
        try:
            if isinstance(message, AIMessage):
                st.write(
                    bot_template
                    .replace("{{MSG}}", message.content)
                    .replace("{{GOOGLE_SEARCH}}", google_search_status)
                    .replace("{{MSG_ID}}", message_id),
                    unsafe_allow_html=True
                )
            elif isinstance(message, HumanMessage):
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.error(f"Unknown message type: {type(message)}")
                return
        except Exception as e:
            st.error(f"Error displaying message: {e}")
            return

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write('<link href="https://cdn.jsdelivr.net/npm/@coreui/coreui@5.0.0/dist/css/coreui-grid.min.css" rel="stylesheet" crossorigin="anonymous">', unsafe_allow_html=True)
    st.write(css, unsafe_allow_html=True)

    session_state = SessionStorage()

    st.header("Chat with multiple PDFs :books:")
    
    # Ensure unique keys
    user_question = st.text_input("Ask a question about your documents:", key="user_input")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True, key="file_uploader")
        if st.button("Process", key="process_button"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)
                st.success("Documents processed successfully.")

if __name__ == "__main__":
    main()