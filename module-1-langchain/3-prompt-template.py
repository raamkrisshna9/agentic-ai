from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm_openai = ChatOpenAI(model="gpt-4.1-nano")
llm_google = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

#from_template is a class method that creates a PromptTemplate instance from a simple string template.
template = PromptTemplate.from_template( """ You are a helpful funny assistant answer the user question using below context, if you are not able to answer based on context, say i dont know.
    Context: {context}
    Question: {question}
    Answer in a humorous way.""")

template_values = {
    "context": "India's capital is New Delhi. It is known for its rich history and cultural heritage.",
    "question": "who is the primeminstare of india?"
}

print ("expected input variables of prompt template:", template.input_variables)

# Example - 1 
template_object = template.invoke(template_values)

# Example - 2
#.format_messages is a method that formats the prompt template into a list of message objects based on the provided input values.
#**template_values unpacks the dictionary to pass its key-value pairs as keyword arguments to the method.
formatted_messages = template.format_messages(**template_values)

print(f"\n--- 3. Execution with format_messages() ---")
print(f"Result Type: {type(formatted_messages)}")
for message in formatted_messages:
    #message.type gives the role of the message (system, user, assistant) and message.content provides the actual text content of the message.
    print(f"[{message.type.upper()}]: {message.content[:]}...")

# Example - 1
response_openai = llm_openai.invoke(template_object)
response_google = llm_google.invoke(template_object)

# Example - 2
response_openai = llm_openai.invoke(formatted_messages)
response_google = llm_google.invoke(formatted_messages)

print("OpenAI Response:")
print("----------------")
print(response_openai.text  , "\n")
print("Google AI Response:")
print("----------------")
print(response_google.text)