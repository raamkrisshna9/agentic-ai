# Use-case workflow:
# - START: entry point for the StateGraph
# - ingest_data: load and store data into the graph state
# - chat: call the LLM to generate responses and update the graph state
# - summarize: condense the state once a message-count threshold is reached

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Optional

load_dotenv()
llm_google = ChatGoogleGenerativeAI(model="gemini-2.5-flash", 
                                    temperature=0.5, 
                                    max_output_tokens=2048)

# Define the graph state
class GraphState(TypedDict):
    """
    prompt: The latest user input message
    messages: Ordered conversational history
    summary: Summary of the messages after reaching to certain threshold
    """
    prompt: str
    messages: list[BaseMessage] #list[BaseMessage] indicates a list where each element is an instance of BaseMessage or its subclasses.
    summary: Optional[str]

# Define the ingest_data node
def ingest_data(state: GraphState) -> GraphState:
    """
    Ingest the latest user input message into the graph state.
    """
    messages  = list(state.get('messages', []))
    messages.append(HumanMessage(content=state['prompt']))
    return {"messages": messages}

# Define the summarize node, used to condense the state so that it doesn't grow indefinitely and result the LLM input size limits and performance issues.
# Define the summarize node, used to condense the state so that it doesn't grow indefinitely and result the LLM input size limits and performance issues.
def summarize(state: GraphState) -> GraphState:
    """
    Summarize the conversational history to condense the state.
    """
    messages = list(state.get('messages', []))
    summary = state.get('summary', "")

    threshold = 2
    if len(messages) < threshold:
        return {"summary": summary}

    else:
        summary_prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at creating concise summaries of conversations."),
            ("human", "Given the following conversation history, provide a concise summary focusing on key facts, preferences, and decisions:\n\n{conversation_history}")
            ])

        # List comprehension to iterate the elements in messages list and format them as required.
        # List comprehension syntax: [operation for item in iterable]
        # e.g: [msg.content for msg in messages]
        # Zip function: Zip function is used to combine two or more iterables (like lists or tuples) element-wise into a single iterable of tuples.
        # e.g: a = [1, 2, 3],  b = ['a', 'b', 'c'], for v1, v2 in zip(a, b): print(v1, v2)
        # Zip comprehension syntax: [(v1, v2) for v1, v2 in zip(a, b)]
        # join() method: The join() method is a string method in Python that takes all items in an iterable (like a list or tuple) and joins them into a single string, with a specified separator.
        # Use the message class name to label messages instead of relying on msg.type which may not exist.
        conversation_text = "\n".join([f"{msg.type}: {msg.content}" for msg in messages])
        prompt = summary_prompt_template.format_prompt(conversation_history=conversation_text).to_messages()
        response = llm_google.invoke(prompt)
        new_summary = response.content
        print("Updated Summary:\n", new_summary)
        return {"summary": new_summary}

# Define the chat node
def chat(state: GraphState) -> GraphState:
    """
    Call the LLM to generate a response based on the conversational history.
    """
    messages = list(state.get('messages', []))
    effective_messages: list[BaseMessage] = []
    if state.get('summary'):
        effective_messages.append(SystemMessage(content=f"Summary of previous conversation: {state['summary']}"))
    # append() method just append the list to the existing list as a single element.
    # extend() method appends each element of the list to the existing list at last. i.e the summary is pretededed to the actual latest message.
    effective_messages.extend(messages)
    response = llm_google.invoke(effective_messages)
    messages.append(AIMessage(content=response.content))
    return {"messages": messages}

# Create the state graph
state_graph = StateGraph(GraphState)
state_graph.add_node("ingest_data", ingest_data)
state_graph.add_node("summarize", summarize)
state_graph.add_node("chat", chat)

# Define the edges between nodes
state_graph.add_edge(START, "ingest_data")
state_graph.add_edge("ingest_data", "summarize")
state_graph.add_edge("summarize", "chat")
state_graph.add_edge("chat", END)

# Add memory saver checkpoint to persist the graph state
memory_saver = MemorySaver()
app = state_graph.compile(checkpointer=memory_saver)
app.get_graph().draw_mermaid_png(output_file_path="memory-detailed.png")


# Runner function to interact with the graph application
def runner(thread_id: str):
    print("Chat started! Type 'exit' to stop.\n")
    config = {"configurable": {"thread_id": thread_id}}
    messages = []
    state = {"prompt": "", "messages": [], "summary": ""}

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Chat ended.")
            break

        state["prompt"] = user_input
        result = app.invoke(state, config=config)

        # Update state
        state["messages"] = result.get("messages", state["messages"])
        state["summary"] = result.get("summary", state["summary"])

        # Print the latest AI response only
        ai_responses = [m for m in state["messages"] if isinstance(m, AIMessage)]
        if ai_responses:
            print("AI:", ai_responses[-1].content)
        #print(app.get_state(config=config))  # For debugging: print the current state from the checkpointer
if __name__ == "__main__":
    thread_id = "user-rama"
    runner(thread_id)