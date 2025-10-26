from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import SystemMessage, HumanMessage

load_dotenv()

llm_openai = ChatOpenAI(model="gpt-4.1-nano")
llm_google = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


system_msg = SystemMessage(content="You are a helpful funny assistant that answers the questions in a humorous way.")
human_msg = HumanMessage(content="who is the primeminstare of india?")

#message = [("system", "You are a helpful funny assistant that answers the questions in a humorous way."), ("user", "who is the primeminstare of india?")]


message = [system_msg, human_msg]

response_openai = llm_openai.invoke(message)
response_google = llm_google.invoke(message)

print("OpenAI Response:")
print("----------------")
print(response_openai.text  , "\n")
print("Google AI Response:")
print("----------------")
print(response_google.text)