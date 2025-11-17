from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import argparse

load_dotenv()
llm_google = ChatGoogleGenerativeAI(model="gemini-2.5-flash", 
                                  temperature=0.5,
                                  max_output_tokens=2056)


prompt_template = ChatPromptTemplate.from_messages([("system", "You are simple travel recommender agent provide best travel recommendations based on user given inputs and preferences"),
                                                    ("user", """Provide travel recommendations based on below inputs.
                                                    Destination: {destination}
                                                    Travel days: {travel_days}
                                                    Budget: {budget}
                                                    Traveller type: {traveller_type}
                                                    Follow the below rules while providing recommendations:
                                                    - Opener: 1-2 lines introduction about the destination.
                                                    - Iterary: Day-wise breakdown of activities and places to visit based on travel days, max at 2-3 lines per day.
                                                    - Travel tips: Budget and traveller type specific suggestions and recommendations in bullet points and not exceed 500 words.
                                                    - closing summary: Brief summary of why this destination is worth visiting with in 1-2 lines and places must visit during the particular travel in bullet points
                                                    - use \n as fromatting for line breaks and bullet points to enhance readability.
                                                    - Follow the typicall text practises to ensure the readability and clarity of the response.
                                                    - Ensure the recommendations are practical and tailored to the specified budget and traveller type.""")])


arg_parser = argparse.ArgumentParser(description="Dynamic Prompt Template Input for Travel Recommendations")
arg_parser.add_argument("--destination", type=str, required=True, help="Travel destination")
arg_parser.add_argument("--travel_days", type=int, required=True, help="Number of travel days")
arg_parser.add_argument("--budget", type=str, required=True, help="Travel budget")
arg_parser.add_argument("--traveller_type", type=str, required=True, help="Type of traveller")
args = arg_parser .parse_args()

prompt_template_values = { "destination": args.destination,
                           "travel_days": args.travel_days,
                           "budget": args.budget,
                           "traveller_type": args.traveller_type } 

chain = prompt_template | llm_google
response_google = chain.invoke(prompt_template_values)

print("The travel recomendation:")
print("-------------------------")
print(response_google.text)


# Execution command example:
# python assignment-a.py --destination "Paris" --travel_days 6 --budget "high" --traveller_type "couple"
# python assignment-a.py --destination "visakhapatnam" --travel_days 3 --budget "medium" --traveller_type "family"
# python assignment-a.py --destination "New York" --travel_days 4 --budget "high" --traveller_type "luxury"
# python assignment-a.py --destination "Tiruvanamalai" --travel_days 2 --budget "low" --traveller_type "solo"