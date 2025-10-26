from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm_openai = ChatOpenAI(model="gpt-4.1-nano")
llm_google = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

message = "what is the capital of india?"

response_openai = llm_openai.invoke([{"role": "user", "content": message}])
response_google = llm_google.invoke([{"role": "user", "content": message}])

print("OpenAI Response:")
print("----------------")
print(response_openai.text  , "\n")
print("Google AI Response:")
print("----------------")
print(response_google.text)