# Usecase:
# Implement a product details extracter using dynamic prompt templates and output parser with validation.
# Input: Product name and description text.
# Output: JSON object containing product name, categeory, price_estimage, pros[], cons[].
# Follow the response vlidtion workflow mentioned below.

# Response validation workflow:
# 1. LLM create the resonse.
# 2. Prase the response on JSON parser.
# 3. Validate the parsed response against a schema using Pydantic model.
# 4. If validation fails, use the error message to guide the LLM to correct the response and re-attempt.
# 5. Repeat until a valid response is obtained or a maximum number of attempts is reached.

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import argparse
from pydantic import BaseModel, ValidationError, Field
from typing import Optional, List
from langchain_core.output_parsers import PydanticOutputParser # Used to create the response output schema
import json


load_dotenv()
llm_google = ChatGoogleGenerativeAI(model="gemini-2.5-flash", 
                                  temperature=0.5,
                                  max_output_tokens=1024)


prompt_template = ChatPromptTemplate.from_messages([("system", "you are a productduct detials extracter, extract the product details from the user input and provide the response in json format as per the schema provided"),
                                         ("user", """Extract the product details from the below product description.
                                          Product Name: {product_name}
                                          Product Description: {product_description}
                                          Maximum Retries: {max_retries}
                                          Provide the response in JSON format and follow below rules:
                                          - {format_instructions}
                                          - The response should be a valid JSON object.
                                          - Dont include any explanations, only provide the JSON object as response.""")
                                          ])

arg_parser = argparse.ArgumentParser(description="Dynamic Prompt Template Input")
arg_parser.add_argument("--product_name", type=str, required=True, help="Name of the product")
arg_parser.add_argument("--product_description", type=str, required=True, help="Description of the product")
arg_parser.add_argument("--max_retries", type=int, required=False, default=3, help="Maximum number of retries for validation")
args = arg_parser .parse_args()

class ProductDetails(BaseModel):
    product_name: str = Field(..., description="Name of the product", min_length=1)
    category: str = Field(..., categeory="Categery of the product", min_length=1)
    price_estimate: Optional[float] = Field(None, description="price estimation of the product", gt=0)
    pros: List[str] = Field(default=list, min_items=1, description="List of pros of the product")
    cons: List[str] = Field(default=list, min_items=1, description="List of cons of the product")

output_parser = PydanticOutputParser(pydantic_object=ProductDetails)
format_instructions = output_parser.get_format_instructions()
print("Format Instructions:")
print("-------------------")
print(format_instructions)

prompt_template_values = { "product_name": args.product_name,
                           "product_description": args.product_description,
                           "max_retries": args.max_retries,
                           "format_instructions": format_instructions }

chain = prompt_template | llm_google

def extract_product_details(chain, prompt_template_values, output_parser, max_retries):
    retries = 0
    prompt_values = dict(prompt_template_values)
    prompt_values.setdefault("validation_error", "")
    
    while retries < max_retries:
        response = chain.invoke(prompt_values)
        try:
            parsed_output = output_parser.parse(response.text)
            return parsed_output
        except (ValidationError, json.JSONDecodeError) as e:
            error_msg = str(e)
            print(f"Validation error: {error_msg}. Retrying... ({retries+1}/{max_retries})")
            retries += 1
            prompt_values["validation_error"] = error_msg
    raise Exception("Maximum retries reached. Could not extract valid product details.")


product_details = extract_product_details(chain, prompt_template_values, output_parser, args.max_retries)
print("Extracted Product Details:")
print("-------------------------")
print(product_details.model_dump_json(indent=2))


# python 7-dynamic-prompt-templates-outputparser.py --product_name "ihone 17" --product_description "The Smartphone iphone 17 features a stunning display, powerful processor, and long-lasting battery life. It offers excellent camera quality and a sleek design. However, it lacks expandable storage and has a higher price point compared to competitors." --max_retries 3