from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

#Loads the environment variables from a .env file into the system's environment variables.
load_dotenv() 

#Defines a language model instance using OpenAI's GPT-4.1-nano model.
llm_openai = ChatOpenAI(model="gpt-4.1-nano")
#Defines a language model instance using Google's Gemini 2.5 Flash model.
llm_google = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

message = "what is the capital of india?"

#Invoke method is called on the llm_openai instance to generate a response based on the provided user message.
response_openai = llm_openai.invoke([{"role": "user", "content": message}])
response_google = llm_google.invoke([{"role": "user", "content": message}])

print("OpenAI Response:")
print("----------------")
print(response_openai.text  , "\n")
print("Google AI Response:")
print("----------------")
print(response_google.text)