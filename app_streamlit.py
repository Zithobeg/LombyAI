import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
import pickle


from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain.document_loaders import DirectoryLoader
import bs4



#-----------------------------------------------------------------------------------------------------------------------------#
#                                                       Vector Store Creation

loader = DirectoryLoader('Documents')
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    #separator="\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)
split_documents  = text_splitter.split_documents(documents)

vectorstore_2 = FAISS.from_documents(split_documents, HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))

with open("vectorstore_2.pkl", "wb") as f:
    pickle.dump(vectorstore_2, f)

#-----------------------------------------------------------------------------------------------------------------------------#



# Page configuration
st.set_page_config(page_title="Lomby AI", layout="wide")

# CSS for chat design
css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 100%;
  padding: 0 1.5rem;
  color: #fff;
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.ibb.co/BVxBFDC/Lombard-Logo.png"  style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://www.freeiconspng.com/uploads/grab-vector-graphic-person-icon--imagebasket-13.png" width="350" alt="Grab Vector Graphic Person Icon | imagebasket" />
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''

# Inject CSS into the app
st.markdown(css, unsafe_allow_html=True)

# Initialize the language model
llm = ChatGroq(temperature=0, model_name="llama3-8b-8192", api_key="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")

# Load retriever function
@st.cache_resource
def load_retriever():
    with open("vectorstore_2.pkl", "rb") as f:
        vectorstore = pickle.load(f)
    return vectorstore

vectorstore_hf = load_retriever()

# Set up the conversational chain
chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=vectorstore_hf.as_retriever(),
    return_source_documents=True,
)

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Function to handle user input
def handle_input(user_input):
    if not user_input.strip():
        return
    
    # Format chat history for the chain input
    formatted_chat_history = [
        (entry["message"], next_entry["message"])
        for entry, next_entry in zip(st.session_state.chat_history[::2], st.session_state.chat_history[1::2])
        if entry["role"] == "user" and next_entry["role"] == "bot"
    ]

    # Query the chain with formatted history
    result = chain({"question": user_input, "chat_history": formatted_chat_history})
    answer = result["answer"]
    sources = result.get("source_documents", [])
    
    # Append the user and bot messages to chat history
    st.session_state.chat_history.append({"role": "user", "message": user_input})
    st.session_state.chat_history.append({"role": "bot", "message": answer, "sources": sources})

# Display chat messages in the interface
for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        st.markdown(user_template.replace("{{MSG}}", chat["message"]), unsafe_allow_html=True)
    elif chat["role"] == "bot":
        st.markdown(bot_template.replace("{{MSG}}", chat["message"]), unsafe_allow_html=True)
        # Show sources in an expandable section
        if "sources" in chat and chat["sources"]:
            with st.expander("View Sources"):
                for source in chat["sources"]:
                    st.write(f"- {source.metadata.get('source', 'Unknown')}")

# Input box and buttons at the bottom
st.divider()  # Adds a visual separation before the input
user_input = st.text_input("Type your message:", key="user_input", label_visibility="hidden", placeholder="Ask Lomby AI...")
col1, col2 = st.columns([3, 1])  # Arrange input and buttons horizontally
with col1:
    pass  # Reserved for input space
with col2:
    if st.button("Send"):
        handle_input(user_input)
        st.experimental_rerun()

# Reset chat button in sidebar
if st.sidebar.button("Reset Chat"):
    st.session_state.chat_history = []

