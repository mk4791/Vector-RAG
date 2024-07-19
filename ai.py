from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from dotenv import load_dotenv
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv()
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
persist_directory = "./db/LionBat/" 
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

model = ChatGoogleGenerativeAI(
    model="gemini-pro",
    temperature=0.3,
    convert_system_message_to_human=True
)

bot_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 2000,
  "response_mime_type": "text/plain",
}

bot = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=bot_config,
  system_instruction=""" your name is BOT , 
  dont use * or any markdown return only plain text,
you will be provided a template for question and extra info in this manner
prompt = {question} extra_data = {hint}
DO NOT GENERATE A JSON type format
only reply the text :
you ahave to reply to that prompt you can use exta_data to help yourself ,
give a proper plane text reply not so long , !!remember!! you are not an llm and you dont provide any other information out of your domain
dont act as an ai act as an character names BOT the instructor with some personality .. dont use symbols like ** `` etc""",
)

vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
print("Vector DB Loaded\n")

chat_session = bot.start_chat(
  history=[
  ]
)


template = """ Use the following pieces of context to answer the question at the end.try to give as much information as u can from the context If you don't know the answer send '...'.
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

query_chain = RetrievalQA.from_chain_type(
    llm=model,
    retriever=vectordb.as_retriever(search_kwargs={"k":5}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

def chat():
    print("Hello! I am your instructor. How can I help you today?")
    while True:
        prompt = input("You: ")
        
        if prompt in ["exit", "quit", "bye"]:
            print("BOT: Goodbye!")
            break
        elif prompt == "hi":
            print("BOT: Hello! How can I assist you today?")
        else:
            hint = query_chain({"query": prompt})
            print("BOT:",hint["result"])
            # response = chat_session.send_message(f"question = {prompt} , extra_data = {hint["result"]}")
            response = chat_session.send_message("question = " + prompt + " , extra_data = " + hint["result"])
            print("BOT:",response.text)


if __name__ == "__main__":
    chat()