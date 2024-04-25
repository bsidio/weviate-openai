from langchain.vectorstores.weaviate import Weaviate
from langchain.llms import OpenAI
from langchain.chains import ChatVectorDBChain
import weaviate

client = weaviate.Client("http://10.0.0.109:8099")

vectorstore = Weaviate(client, "PodClip", "content")

MyOpenAI = OpenAI(temperature=0.2,
    openai_api_key="")

qa = ChatVectorDBChain.from_llm(MyOpenAI, vectorstore)

chat_history = []

print("Welcome to the Weaviate ChatVectorDBChain Demo!")
print("Please enter a question or dialogue to get started!")

while True:
    query = input("")
    result = qa({"question": query, "chat_history": chat_history})
    print(result["answer"])
    chat_history = [(query, result["answer"])]