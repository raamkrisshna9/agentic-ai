from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm_google = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


# Example - 1
template = ChatPromptTemplate.from_template( """ You are a helpful funny assistant answer the user question using below context, if you are not able to answer based on context, say i dont know.
    Context: {context}
    Question: {question}
    Answer in a humorous way.""")

template_values = {
    "context": "India's capital is New Delhi. It is known for its rich history and cultural heritage.",
    "question": "who is the primeminstare of india?"
} 

# Example - 2
template = ChatPromptTemplate.from_messages([("system", "You are a helpful funny assistant answer the user question using below context, if you are not able to answer based on context in humerous way, say i dont know"),
                                             ("user", "Context: {context}\nQuestion: {question}")])

template_values = {
    "context": "India's capital is New Delhi. It is known for its rich history and cultural heritage.",
    "question": "who is the primeminstare of india?"
}

chain = template | llm_google
response_google = chain.invoke(template_values)

print("Google AI Response:")
print("----------------")
print(response_google.text)




