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
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import TypedDict, Optional
import sqlite3
import re

load_dotenv()
llm_google = ChatGoogleGenerativeAI(model="gemini-2.5-flash", 
                                    temperature=0.1, 
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
        # Filter out SystemMessages from the conversation history for summarization
        filtered_messages = [ msg for msg in messages if not (isinstance(msg, SystemMessage) and msg.additional_kwargs and msg.additional_kwargs.get("marker") == "first_time_login")]

        # List comprehension to iterate the elements in messages list and format them as required.
        # List comprehension syntax: [operation for item in iterable]
        # e.g: [msg.content for msg in messages]
        # Zip function: Zip function is used to combine two or more iterables (like lists or tuples) element-wise into a single iterable of tuples.
        # e.g: a = [1, 2, 3],  b = ['a', 'b', 'c'], for v1, v2 in zip(a, b): print(v1, v2)
        # Zip comprehension syntax: [(v1, v2) for v1, v2 in zip(a, b)]
        # join() method: The join() method is a string method in Python that takes all items in an iterable (like a list or tuple) and joins them into a single string, with a specified separator.
        # Use the message class name to label messages instead of relying on msg.type which may not exist.
        conversation_text = "\n".join([f"{msg.type}: {msg.content}" for msg in filtered_messages])
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
        effective_messages.append(SystemMessage(content=f"Answer precisely and concisely. Summary of previous conversation: {state['summary']}"))
    
    # Filter out first_time SystemMessages from the conversation history for LLM input
    filtered_messages = [ msg for msg in messages if not ( isinstance(msg, SystemMessage) and msg.additional_kwargs and msg.additional_kwargs.get("marker") == "first_time_login" )]
    # append() method just append the list to the existing list as a single element.
    # extend() method appends each element of the list to the existing list at last. i.e the summary is pretededed to the actual latest message.
    effective_messages.extend(filtered_messages)
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


# Runner function to interact with the graph application
def runner(thread_id: str, is_new_user: bool, user_name: str, user_dob: str):
    config = {"configurable": {"thread_id": thread_id}}

    # Retrieve saved state from checkpointer and assign to current state
    saved_state = app.get_state(config=config)
    state = {
        "prompt": "",
        "messages": saved_state.values.get("messages", []),
        "summary": saved_state.values.get("summary", "")
    }

    # If new user, add first time login marker message
    if is_new_user:
        first_time_msg = SystemMessage(
            content=f"First time login: {user_name}-{user_dob}",
            additional_kwargs={"marker": "first_time_login"}
        )
        # Insert the first time login marker message at the beginning of the messages list
        state["messages"].insert(0, first_time_msg)
        print(f"Welcome {user_name.capitalize()}! Your profile has been created.\n")
        print("Chat started! Type 'exit' to stop.\n")

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


def validate_name(user_name: str):
    errors = []
    if not user_name:
        errors.append("First name is required.")
    if " " in user_name or "-" in user_name:
        errors.append("Name must not contain spaces or hyphens.")
    if not user_name.isalpha():
        errors.append("Name must contain only alphabetic characters.")
    if len(user_name) < 2:
        errors.append("Name must be at least 2 characters long.")
    clean_name = user_name.lower().strip()
    return errors, clean_name

def validate_dob(user_dob: str):
    errors = []
    dob_pattern = r"^\d{2}-\d{2}-\d{4}$"
    if not re.match(dob_pattern, user_dob):
        errors.append("DOB must be in strict DD-MM-YYYY format (e.g. 01-12-1990).")
    else:
        day = int(user_dob[:2])
        month = int(user_dob[3:5])
        year = int(user_dob[6:])
        if not (1 <= day <= 31):
            errors.append("Day (DD) must be between 01 and 31.")
        if not (1 <= month <= 12):
            errors.append("Month (MM) must be between 01 and 12.")
        if not (1900 <= year <= 9999):
            errors.append("Year (YYYY) must be between 1900 and 9999.")
    return errors, user_dob

if __name__ == "__main__":
    # Initialize SQLite checkpointer
    conn = sqlite3.connect("checkpoint.db", check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    app = state_graph.compile(checkpointer=checkpointer)

    # Validate name
    while True:
        user_name_input = input("Enter your first name: ").strip()
        name_errors, user_name = validate_name(user_name_input)
        if name_errors:
            for err in name_errors:
                print(err)
            print("Please try again.\n")
        else:
            break  # Name is valid

    #Validate DOB
    while True:
        user_dob_input = input("Enter your date of birth (DD-MM-YYYY): ").strip()
        dob_errors, user_dob = validate_dob(user_dob_input)
        if dob_errors:
            for err in dob_errors:
                print(err)
            print("Please try again.\n")
        else:
            break  # DOB is valid

    # Formation of thread ID based on user details
    user_identifier = f"{user_name}-{user_dob}"
    thread_id = f"user-{user_identifier}"
    config = {"configurable": {"thread_id": thread_id}}

    # Retrieve saved state from checkpointer
    existing_state = app.get_state(config=config)
    existing_messages = existing_state.values.get("messages", [])

    # Check for first time login marker
    has_first_login_marker = any(
        isinstance(msg, SystemMessage)
        and msg.additional_kwargs
        and msg.additional_kwargs.get("marker") == "first_time_login"
        for msg in existing_messages
    )

    # Real conversation check
    existing_message_count = len([
        m for m in existing_messages
        if isinstance(m, (HumanMessage, AIMessage))
    ])

    # Greet user based on profile status
    if has_first_login_marker or existing_message_count > 0:
        print(f"\nWelcome back {user_name.capitalize()} (Your DOB: {user_dob})!")
        print("We found your previous session. Do you want to continue? (yes/no)")
        conf = input("> ").strip().lower()

        if conf not in ["yes", "y"]:
            print("Okay! Your session is saved. Goodbye!")
            exit(0)

        print("Chat started! Type 'exit' to stop.\n")
        is_new_user = False
    else:
        print(f"\nThis looks like your first time here.")
        is_new_user = True

    # Start the chat runner
    runner(thread_id, is_new_user, user_name, user_dob)