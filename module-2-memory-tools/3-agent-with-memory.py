from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict
from argparse import ArgumentParser
import argparse

load_dotenv()
llm_google = ChatGoogleGenerativeAI(model="gemini-2.5-flash", 
                                    temperature=0.5, 
                                    max_output_tokens=2048)

class GraphState(TypedDict):
    input: str
    response: str

def llm_node(state: GraphState): 
    prompt_values = state.get("input", "")
    response = llm_google.invoke(prompt_values)
    return {"response": response.content}

graph = StateGraph(GraphState)
graph.add_node("llm", llm_node)
graph.add_edge(START, "llm")
graph.add_edge("llm", END)
# Define a MemorySaver  to enable memory checkpointing for the graph application. (Uses the InMemoeyStore to store the checkpoints)
checkpointer = MemorySaver()
# Compile the graph into an executable application with the checkpointer.
app = graph.compile(checkpointer=checkpointer)
app.get_graph().draw_mermaid_png(output_file_path="hello-langgraph.png")

arg_parser = argparse.ArgumentParser(description="Dynamic prompt Input")
arg_parser.add_argument("--input", type=str, required=True, help="Question to ask the LLM")
args = arg_parser.parse_args()

prompt = {"input": args.input}
# Define configuration settings for the checkpointer. 
config = {"configurable" : {"thread_id": 1}} #The thread_id can be any custom string.
result = app.invoke(prompt, config=config) # Thread_id should be passed during invocation to identify the memory checkpoint.

print("Response:\n", result["response"])

# Execution command example:
# python 2-hello-langgraph.py --input "Explain quantum computing in simple words for the beginners in less than 1000 words"