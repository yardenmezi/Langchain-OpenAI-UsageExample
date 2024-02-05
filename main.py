import os
from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


from langchain.indexes.vectorstore import VectorstoreIndexCreator
load_dotenv()


def useOpenAi(query: str):
    os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")
    info = TextLoader('data.txt')

    index = VectorstoreIndexCreator().index.from_loaders([info])
    print("****** The query result from the loaded data and chat GPT *****\n" + index.query(query, llm=ChatOpenAI()))


if __name__ == '__main__':
    useOpenAi(input("What is your query today?"))
