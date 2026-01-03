from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import WebBaseLoader

load_dotenv()
llm_google = ChatGoogleGenerativeAI(model="gemini-2.5-flash", 
                                    temperature=0.5, 
                                    max_output_tokens=2048)
# Initialise Web Loader
web_loader = WebBaseLoader("https://docs.langchain.com/oss/python/integrations/document_loaders/web_base")

documents = web_loader.load()
print(documents)
print(documents[0].metadata)  # Print metadata of the first Document object
print(documents[0].page_content)  # Print page content of the first Document object
print(documents[0].metadata['source'])  # Prints source URL of the first Document object