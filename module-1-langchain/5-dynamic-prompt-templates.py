from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
llm_google = ChatGoogleGenerativeAI(model="gemini-2.5-flash", 
                                  temperature=0.5,
                                  max_output_tokens=1024)

prompt_template = ChatPromptTemplate.from_messages([("system", "You are a exprt assistant answer the user question in short and crisp manner and alos restrict the short and clean answers"),
                                         ("user", """what is {topic}, explain in {style} for the {audiance}
                                          Answer in {length}
                                          start: Neat and clean
                                          Body: Detailed explanation
                                          End: Summary""")
                                          ])
# Typically system message is used to set the behavior of the assistant, while user message contains the actual prompt with placeholders.
# In general system message not have variable placeholders, only user message has variable placeholders.

prompt_template_values = { "topic": "Quantum Computing",
                  "style": "simple words",
                  "audiance": "post graduate students",
                  "length": "in less than 1000 words"}  

# Prompt_values should be a dictionary containing the values for the placeholders defined in the prompt template. 
# max_output_tokens: The maximum number of tokens to generate in the output at LLM level. Adjust this based on your expected response length.
# The max_output_tokens and length mentioned in user message were works at different levels to control the respse length.
# max_output_tokens says max tokens used in any instance of of response.
# length variable in user message says a instance response lengh

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

# The response_metadata attribute provides additional information about the response, such as token usage, latency, etc used for debugging purposes.


# Note: In this example the prompt template values were hardcoded, we can pass the values dynamically based on user input or other sources as needed uing argparse python package.

