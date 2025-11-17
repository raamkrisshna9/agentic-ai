# Use-case workflow:
# will implement the tools caclulater and web-search as nodes in the StateGraph

from dotenv import load_dotenv
from typing import Annotated, Optional, TypedDict, List
import numexpr
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langchain_core.tools import tool

load_dotenv()

# Define calculator tool node
@tool
def calculator(query: str) -> str:
    """Evaluate a mathematical expression and return the result as a string."""
    try:
        result = numexpr.evaluate(query)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"

# Define search tool
search_tool = DuckDuckGoSearchRun()
# Wrap the tools into a list
tools = [calculator, search_tool]

#llm_google = ChatGoogleGenerativeAI(model="gemini-2.5-flash", 
#                                    temperature=0.1, 
#                                    max_output_tokens=2048).bind_tools(tools)

llm_openai = ChatOpenAI(model="gpt-4-0613",
                        temperature=0.1).bind_tools(tools)

# Define the graph state
class GraphState(TypedDict):
    """
    messages: The given user input message
    """
    #Annotated[list, add_messages] indicates that the messages field is a list that should be processed using the add_messages function.
    #add_messages is likely a function that helps in managing or formatting the messages when they are added to the graph state.
    #In this case we used add_messages instead a manual .append() in the ingest_data node and chat node. Both do the same job.
    messages: Annotated[list, add_messages]

# Define the LLM node
def llm_node(state: GraphState) -> GraphState:
    """ Call the LLM with the current messages and return the updated messages.
    """
    prompt = state.get("messages", [])
    response = llm_openai.invoke(prompt)
    return {"messages": response}


graph = StateGraph(GraphState)
graph.add_node("llm", llm_node)
graph.add_node("tools" , ToolNode(tools))
graph.add_edge(START, "llm")
graph.add_conditional_edges("llm", tools_condition)
graph.add_edge("tools", "llm")
graph.add_edge("llm", END)

app = graph.compile()
app.get_graph().draw_mermaid_png(output_file_path="agent-with-tools.png")

# Example -1
input = {"messages": [HumanMessage(content="calculate the square root of 256 and search for its history on the web.")]}
result = app.invoke(input)
print("-------------------------")
print("Agent final response:\n", result)
print("-------------------------")

# Example -2
#input = {"messages": [HumanMessage(content="Calculate the Fixed deposit interest for 100000 INR for 1 year at the current RBI repo rate?")]}
#result = app.invoke(input)
#print("-------------------------")
#print("Agent final response:\n", result)
#print("-------------------------")

for chunk in graph.stream(input):
    print(chunk)

# Example output-1:
# ================
#{'messages': [HumanMessage(content='calculate the square root of 256 and search for its history on the web.', 
#                           additional_kwargs={}, 
#                           response_metadata={}, 
#                           id='80906210-9ba4-43ce-958d-3e4b37e6250a'
#                           ), 
#              AIMessage(content='', 
#                        additional_kwargs={'function_call': {'name': 'calculator', 'arguments': '{"query": "sqrt(256)"}'}}, 
#                        response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'model_name': 'gemini-2.5-flash', 'safety_ratings': [], 'model_provider': 'google_genai'}, 
#                        id='lc_run--46aa3c30-8a06-4112-8133-ad9e83350b34-0', 
#                        tool_calls=[{'name': 'calculator', 'args': {'query': 'sqrt(256)'}, 'id': '7b38c64e-ec5f-4fc3-a3f2-52dc49752c16', 'type': 'tool_call'}], 
#                        usage_metadata={'input_tokens': 126, 'output_tokens': 94, 'total_tokens': 220, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 77}}
#                        ), 
#              ToolMessage(content='16.0', 
#                          name='calculator', 
#                          id='d2d70c4d-50c8-4a38-aa85-7274546a99ec', 
#                          tool_call_id='7b38c64e-ec5f-4fc3-a3f2-52dc49752c16'
#                          ),
#               AIMessage(content='', 
#                         additional_kwargs={'function_call': {'name': 'duckduckgo_search', 'arguments': '{"query": "history of the number 16"}'}}, 
#                         response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'model_name': 'gemini-2.5-flash', 'safety_ratings': [], 'model_provider': 'google_genai'}, 
#                         id='lc_run--90f4d047-0553-40e3-a830-2ba6ea58346f-0', 
#                         tool_calls=[{'name': 'duckduckgo_search', 'args': {'query': 'history of the number 16'}, 'id': 'c1ea19e9-b7ed-41b8-a34e-ed737fddedf8', 'type': 'tool_call'}], 
#                         usage_metadata={'input_tokens': 157, 'output_tokens': 23, 'total_tokens': 180, 'input_token_details': {'cache_read': 0}}
#                         ), 
#               ToolMessage(content='Number systems have progressed from the use of fingers and tally marks, perhaps more than 40,000 years ago, to the use of sets of glyphs able to represent any conceivable number efficiently. The earliest known unambiguous notations for numbers emerge... The Simple English Wiktionary has a definition for: sixteen . 16 is a number . It comes between fifteen and seventeen, and is an even number . It is also the 4th square number , after 1, 4, and 9. It is the base of hexadecimal numbers . In Roman numerals,... The numerals 7. 16 .6. 16 .18 translate to September 32 B.C.E. (Julian). The glyphs surrounding the date are what is thought to be one of the few surviving examples of Epi-Olmec script. History of zero.The Universal History of Numbers : From Prehistory to the Invention of the Computer. Throughout history , numbers have held great significance and meaning. Number 16716 carries powerful vibrations in the realms of love, money, symbolism, and relationships. Sixteen – Sixteenth. The main exceptions are with the numbers 1, 2, and 3. one – first. two – second.Louis XVI (Louis the Sixteenth) was the last king of France before the fall of the monarchy during the French Revolution. List of Ordinal Numbers .', 
#                           name='duckduckgo_search', 
#                           id='5116c331-6dec-49d5-a434-d6995022b1bf', 
#                           tool_call_id='c1ea19e9-b7ed-41b8-a34e-ed737fddedf8'
#                           ), 
#               AIMessage(content="The square root of 256 is 16. The number 16 has a rich history, appearing in various contexts such as number systems (it's the base of hexadecimal numbers), as the 4th square number, and even in historical figures like Louis XVI.", 
#                         additional_kwargs={}, 
#                         response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'model_name': 'gemini-2.5-flash', 'safety_ratings': [], 'model_provider': 'google_genai'}, 
#                         id='lc_run--9cd8ea90-2238-4c91-ba5d-e84d260e8bc6-0', 
#                         usage_metadata={'input_tokens': 486, 'output_tokens': 58, 'total_tokens': 544, 'input_token_details': {'cache_read': 0}})
#            ]
#} 

# Example output-2:
# ================
#{'messages': [HumanMessage(content='Calculate the Fixed deposit interest for 100000 INR for 1 year at the current RBI repo rate?', 
#                           additional_kwargs={}, 
#                           response_metadata={}, 
#                           id='60b8c94c-2abb-4848-bfb8-b3dbec51a68c'
#                           ), 
#              AIMessage(content='', 
#                        additional_kwargs={'function_call': {'name': 'duckduckgo_search', 'arguments': '{"query": "current RBI repo rate"}'}}, 
#                        response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'model_name': 'gemini-2.5-flash', 'safety_ratings': [], 'model_provider': 'google_genai'}, 
#                        id='lc_run--d983b8e9-d4f6-4f97-91e3-efc908be947c-0', 
#                        tool_calls=[{'name': 'duckduckgo_search', 'args': {'query': 'current RBI repo rate'}, 'id': '80c2f549-106a-4ec4-ba64-42d1e39fc515', 'type': 'tool_call'}], 
#                        usage_metadata={'input_tokens': 133, 'output_tokens': 84, 'total_tokens': 217, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 64}}
#                        ), 
#              ToolMessage(content='Jun 6, 2025 · The Repo Rate (stands for ‘Repurchase Agreement or Repurchasing Option’) is the interest rate at which the RBI (Reserve Bank of India) lends money to commercial banks in … Oct 1, 2025 · RBI Policy today, New RBI Rates 1 October, 2025 : SLR 18.00%, CRR is 3.75%, MSF is 5.75%, Repo Rate is: 5.50%, Reverse Repo Rate is 3.35%, and Bank Rate 5.75%. Updated RBI … Oct 1, 2025 · RBI MPC Meeting 2025 Highlights: Repo rate unchanged at 5.5%, neutral stance continues The Reserve Bank of India (RBI) on Wednesday kept its policy interest rate unchanged … Aug 6, 2025 · The Reserve Bank of India (RBI) lends money to Commercial Banks at a Repo Rate, helping banks maintain their liquidity in case of a shortage of funds or meeting regulatory … Oct 1, 2025 · The RBI has cut the repo rate by 1% or 100 basis points this year from 6.5% to 5.5%, bringing down EMIs for the common man.', 
#                          name='duckduckgo_search', 
#                          id='468cc7a5-147e-4c91-bda8-aa080aaa7802', 
#                          tool_call_id='80c2f549-106a-4ec4-ba64-42d1e39fc515'
#                          ), 
#              AIMessage(content='', 
#                        additional_kwargs={'function_call': {'name': 'calculator', 'arguments': '{"query": "100000 * 0.055 * 1"}'}}, 
#                        response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'model_name': 'gemini-2.5-flash', 'safety_ratings': [], 'model_provider': 'google_genai'}, 
#                        id='lc_run--d2096662-c228-4805-b2e3-240e2e161ad5-0', 
#                        tool_calls=[{'name': 'calculator', 'args': {'query': '100000 * 0.055 * 1'}, 'id': '11e4e701-4c97-4e55-8389-04363842601c', 'type': 'tool_call'}], 
#                        usage_metadata={'input_tokens': 445, 'output_tokens': 28, 'total_tokens': 473, 'input_token_details': {'cache_read': 0}}
#                        ), 
#              ToolMessage(content='5500.0', 
#                          name='calculator', 
#                          id='3ea12ef3-5f17-4873-adbe-b844b35a120c', 
#                          tool_call_id='11e4e701-4c97-4e55-8389-04363842601c'
#                          ), 
#              AIMessage(content='The fixed deposit interest for 100000 INR for 1 year at the current RBI repo rate of 5.5% would be 5500 INR.', 
#                        additional_kwargs={}, 
#                        response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'model_name': 'gemini-2.5-flash', 'safety_ratings': [], 'model_provider': 'google_genai'}, 
#                        id='lc_run--988ae9c4-a3a3-4702-a592-e11dbcf634b9-0', 
#                        usage_metadata={'input_tokens': 489, 'output_tokens': 38, 'total_tokens': 527, 'input_token_details': {'cache_read': 0}}
#                        )
#              ]
#}

# Key points:
# ==========
# LLM is call AIMessage having the tool_calls filed (list of dict) mentioning tool name, args[query] and additional_kwargs capture the function_call details.
# The tool execution result is stored in ToolMessage with content as the tool output.
# The LLM is calling one tool after another based on the user query until it reaches the final answer.
# The final response is in the last AIMessage in the messages list.
