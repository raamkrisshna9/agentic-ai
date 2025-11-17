from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, ValidationError
from typing import Optional
from langchain_core.output_parsers import PydanticOutputParser
import argparse, json,  re

load_dotenv()
llm_google = ChatGoogleGenerativeAI(model="gemini-2.5-flash", 
                                  temperature=0.1,
                                  max_output_tokens=2056)

llm_openai = ChatOpenAI(model="gpt-4.1-nano",
                        temperature=0.1,
                        max_tokens=2056)


prompt_template = ChatPromptTemplate.from_messages([("system", "You are a restaurant review analyzer and fields extracting agent, extract the required fields from the user given restaurant review and provide the response in json format as per the schema provided"),
                                            ("user", """Extract the restaurant details given in the format instructions from the below restaurant review.
                                                Restaurant Review: {resturant_review}
                                                Provide the response in JSON format and follow below rules:
                                                - {format_instructions}
                                                - The response should be ONLY a valid JSON object.
                                                - Do not include any explanations and fabricate facts, If data is missing.""")])

arg_parser = argparse.ArgumentParser(description="Dynamic Prompt Template Input for Resturant Review Analysis")
arg_parser.add_argument("--resturant_review", type=str, required=True, help="Resturant review text")
args = arg_parser .parse_args()

max_retries = 5

class ResturantDetails(BaseModel):
    name: str = Field(..., description="Name of the resturant", min_length=1)
    cuisine: str = Field(..., description="Type of cuisine served, e.g: indian, italian, japan etc", min_length=1)
    city: str = Field(default="empty", description="city of the resturant", min_length=1)
    rating: Optional[float] = Field(None, description="Resturant rating out of 5", ge=0.0, le=5.0)
    price_range: str = Field(..., description="Price range like low, mild, High", choices=["low", "mid", "high"])

output_parser = PydanticOutputParser(pydantic_object=ResturantDetails)
format_instructions = output_parser.get_format_instructions()

prompt_template_values = { "resturant_review": args.resturant_review,
                           "format_instructions": format_instructions }

chain = prompt_template | llm_openai

def extract_resturant_details(chain, prompt_template_values, output_parser, max_retries):
    def clean_response_text(text: str) -> str:
        if not isinstance(text, str):
            return text
        # Trim whitespace and BOM
        text = text.strip().lstrip("\ufeff").rstrip()
        # Remove leading triple-backtick fence with optional language tag
        text = re.sub(r'^\s*```[^\n]*\n?', '', text)
        # Remove trailing triple-backtick fence
        text = re.sub(r'\n?\s*```\s*$', '', text)
        # Remove single backticks if they wrap the entire content
        if text.startswith('`') and text.endswith('`'):
            text = text.strip('`').strip()
        return text

    retries = 0
    prompt_values = dict(prompt_template_values)
    prompt_values.setdefault("validation_error", "")

    while retries < max_retries:
        response = chain.invoke(prompt_values)
        raw_text = response.content if hasattr(response, "content") else (response.text if hasattr(response, "text") else str(response))
        cleaned_text = clean_response_text(raw_text)

        try:
            print("\nAttempting to parse response...")
            print("Try loop number:", retries + 1)
            print("----------------------")
            if not cleaned_text or cleaned_text.strip() == "":
                raise ValueError("Empty response from LLM")
            
            parsed_output = output_parser.parse(cleaned_text)
            return parsed_output
        except (ValidationError, json.JSONDecodeError, ValueError) as e:
            error_msg = str(e)
            print(f"Validation error: {error_msg}. Retrying... ({retries+1}/{max_retries})")
            retries += 1
            prompt_values["validation_error"] = error_msg
    raise Exception("Maximum retries reached. Could not extract valid product details.")

resturant_details = extract_resturant_details(chain, prompt_template_values, output_parser, max_retries)
print("Extracted Product Details:")
print("-------------------------")
print(resturant_details.model_dump_json(indent=2))


# python assignment-b.py --resturant_review "I recently visited The Spice House in New York and had an amazing experience. The Indian cuisine was authentic and flavorful, with dishes like butter chicken and biryani standing out. The ambiance was cozy, and the staff were friendly and attentive. I would rate it 4.5 out of 5. The prices were reasonable for the quality of food and service provided. Overall, a must-visit for anyone craving delicious Indian food!"
