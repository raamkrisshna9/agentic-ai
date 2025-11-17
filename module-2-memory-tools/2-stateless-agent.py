from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()
llm_google = ChatGoogleGenerativeAI(model="gemini-2.5-flash", 
                                    temperature=0.5, 
                                    max_output_tokens=1024)

prompt = ChatPromptTemplate.from_messages([("system", "You are a helpful concise assiant that provides brief answers."),
    ("user", "{input}?")])

chain = prompt | llm_google

while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        print("Exiting the conversation.")
        break

    prompt_values = {"input": user_input}
    response_google = chain.invoke(prompt_values)
    print(f"Bot:{response_google.content}")

# Example Interaction:
# python 1-stateless-agent.py

# You: Iam plaining for a trip to dubai
# Bot:Great! To help you best, what kind of information are you looking for (e.g., best time to visit, things to do, visa, budget)?

# You: Get me the flight tickets for it?
# Bot:I cannot directly book tickets. Please provide:
# *   **Departure city:**
# *   **Destination city:**
# *   **Departure date:**
# *   **Return date (if applicable):**
# *   **Number of passengers:**
# *   **Preferred airline/time (optional):**

# You: Bangalore, Dubai, 6th Nov 2025, 10th Nov 2025, 2, Indigo 
# Bot:Indigo flights from Bangalore to Dubai for 2 passengers, departing 6th Nov 2025 and returning 10th Nov 2025.


# Note: This is a stateless agent, meaning it does not retain memory of past interactions. Each user input is treated independently.