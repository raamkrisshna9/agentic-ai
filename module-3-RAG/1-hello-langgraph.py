from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from argparse import ArgumentParser
import argparse

load_dotenv()
llm_google = ChatGoogleGenerativeAI(model="gemini-2.5-flash", 
                                    temperature=0.5, 
                                    max_output_tokens=2048)

#The GraphState class can be TypedDict or a Pydantic model to define the schema of the state data used in the graph.
#TypedDict is used to define a dictionary with specific key-value types, providing type hints for better code clarity and error checking.
class GraphState(TypedDict):
    input: str
    response: str

#The function llm_node takes a single argument state of type GraphState as a input parameter.
#i.e Reading the input value from the state dictionary, invoking the llm_google model with that input, and returning a new dictionary containing the response into the state.
def llm_node(state: GraphState): 
    #The get() method is used to retrieve the value associated with the "input" key from the state dictionary.
    #The get() wont raise a valueError as simple dict indexing, if the key is missing, instead it returns an empty string as default, allowed to pass the default value.
    prompt_values = state.get("input", "")
    response = llm_google.invoke(prompt_values)
    return {"response": response.content}

#StateGraph defines the graph structure using the GraphState schema.
graph = StateGraph(GraphState)
#Adding nodes to the graph using add_node method.
graph.add_node("llm", llm_node)
#Adding edges to define the flow between nodes.
graph.add_edge(START, "llm")
graph.add_edge("llm", END)
#Compiling the graph into an executable application.
app = graph.compile()
#Generating a visual representation of the graph and saving it as a PNG file.
app.get_graph().draw_mermaid_png(output_file_path="hello-langgraph.png")

arg_parser = argparse.ArgumentParser(description="Dynamic prompt Input")
arg_parser.add_argument("--input", type=str, required=True, help="Question to ask the LLM")
args = arg_parser.parse_args()

prompt = {"input": args.input}
result = app.invoke(prompt) #Graph need to be invoke only after compilation.

print("Response:\n", result["response"])

# Execution command example:
# python 2-hello-langgraph.py --input "Explain quantum computing in simple words for the beginners in less than 1000 words"