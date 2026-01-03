
################################################
# load_corpus.py
# Load corpus and add to vector store
# This need to be run before running the main.py
################################################

from vector_store import initialize_redis_vector_store, add_documents_to_vector_store

corpus = ["Agentic AI: An advanced AI system capable of autonomously setting sub-goals, reasoning, planning, and executing multi-step actions using various tools to achieve a higher-level objective with minimal human oversight.",
    "AI Agent: A fundamental software entity that continuously perceives its environment, reasons about the information, and takes actions to achieve specific goals. They are the building blocks of agentic AI.",
    "Autonomy: The core characteristic of Agentic AI, referring to the system's ability to operate and make decisions independently without continuous human intervention or step-by-step guidance.",
    "Perception: The process by which an AI agent collects and interprets data, information, or signals from its environment (digital or physical) to understand its current state and context.",
    "Reasoning: The cognitive process, often driven by a Large Language Model (LLM), where the AI agent draws conclusions, makes decisions, and formulates a plan of action based on the perceived information and its overall goal.",
    "Action / Tool Use: The ability of an AI agent to execute a decision by interacting with external systems, databases, APIs, or software tools to perform a task in the real world.",
    "Orchestration: The coordination and management of multiple AI agents, each often specializing in a different role or sub-task, to work collaboratively and synchronously towards a single, complex objective.",
    "Generative AI (GenAI): An AI model focused on the creation of new content (text, images, code) based on input prompts. Agentic AI often uses GenAI models as its 'brain' for reasoning.",
    "Human in the Loop (HITL): A safety and governance practice where an autonomous AI agent is designed to seek human review or approval before executing actions that are classified as high-risk or high-impact.",
    "Adaptability: The agent's ability to dynamically adjust its plans, strategies, and actions based on changing conditions or unexpected outcomes encountered in its environment.",
    "Memory: The mechanism that allows an AI agent to store and retrieve past interactions, states, and context over time to ensure task continuity and improve future decision-making.",
    "Multi-agent System: An architecture involving two or more specialized AI agents that communicate and collaborate under a supervisory agent or orchestration framework to complete a single, large workflow."
]
# Add corpus to Redis vector store
add_documents_to_vector_store(vector_store=initialize_redis_vector_store(), documents=corpus)