from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import argparse

#https://docs.python.org/3/howto/argparse.html

load_dotenv()
llm_google = ChatGoogleGenerativeAI(model="gemini-2.5-flash", 
                                  temperature=0.5,
                                  max_output_tokens=1024)

prompt_template = ChatPromptTemplate.from_messages([("system", "You are a exprt assistant answer the user question in short and crisp manner and alos restrict the short and clean answers"),
                                         ("user", """what is {topic}, explain in {style} for the {audiance}
                                          Answer in {length}""")])

arg_parser = argparse.ArgumentParser(description="Dynamic Prompt Template Input")
arg_parser.add_argument("--topic", type=str, required=True, help="Topic to explain")
arg_parser.add_argument("--style", type=str, required=True, help="Style of explanation")
arg_parser.add_argument("--audiance", type=str, required=True, help="Target audience")
arg_parser.add_argument("--length", type=str, required=True, help="Length of the answer")
args = arg_parser .parse_args()

prompt_template_values = { "topic": args.topic,
                           "style": args.style,
                           "audiance": args.audiance,
                           "length": args.length }

chain = prompt_template | llm_google
response_google = chain.invoke(prompt_template_values)

print("Google AI Response Text:")
print("----------------")
print(response_google.text) #print(response_google.content) both are same
print("Google AI Whole Response:")
print("----------------")
print(response_google)
print("\nResponse Metadata:")
print("----------------")
print(response_google.response_metadata) 


# Execution command example: # python 6-dynamic-prompt-templates-argparse.py --topic "Quantum Computing" --style "simple words" --audiance "post graduate students" --length "in less than 1000 words"


