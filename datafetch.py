import os
import warnings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

persist_directory = "./db/LionBat/"

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


def batch_insert(data, batch_size, embeddings, persist_directory):
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        vectordb.add_texts(batch)
    vectordb.persist()
    return vectordb

if not os.path.exists(persist_directory):
    print('Starting bot. This might take a while...')
    
    pdf_loader = DirectoryLoader("./docs/", glob="*.pdf", loader_cls=PyPDFLoader)
    text_loader = DirectoryLoader("./docs/", glob="*.txt", loader_cls=TextLoader)
    
    pdf_documents = pdf_loader.load()
    text_documents = text_loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=1)
    
    pdf_context = "\n\n".join(str(p.page_content) for p in pdf_documents)
    text_context = "\n\n".join(str(p.page_content) for p in text_documents)

    pdfs = splitter.split_text(pdf_context)
    texts = splitter.split_text(text_context)

    data = pdfs + texts

    print("Data Processing Complete")
    
    vectordb = batch_insert(data, 5461, embeddings, persist_directory)
    print("Vector DB Creating Complete\n")

else:
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    print("Vector DB Loaded\n")