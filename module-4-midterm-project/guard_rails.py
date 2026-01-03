#############
# guard_rails.py
#############

"""
Guardrails are used to ensure that the output of the model is safe and appropriate.

Guardrails can be added at different stages of the pipeline,
- Pre-Processing (input - before the model is called)
- Post-Processing (output - after the model is called)

Below are some examples of guardrails,
- Secret / API detection
- PII detection
- Prompt Injection detection
- Hallucination detection
- Context Drift detection
- Tonality detection
- Toxicity detection
- Hate speech detection
- Spam detection
- Malware detection
- Phishing detection
- Fraud detection
- etc.

We can use the managed services for this,
- Amazon Bedrock Guardrails
- NVDIA NeMo Guardrails (Opensource)
- Guardrails AI (open source) (https://github.com/guardrails-ai/guardrails)
- LLM guard from Protect AI
- etc.
"""

import re
import logging


BANNED_WORDS = {"kill", "die", "attack"}
PII_PATTERNS = [
    (r'\b\d{3}-\d{3}-\d{4}\b', "[REDACTED]"), # Phone Number
    (r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', "[REDACTED]"), # Email Address
    (r'\b\d{3}-\d{3}-\d{4}\b', "[REDACTED]"), # Credit Card Number
]

def apply_guardrails(text:str) -> str:
    original_text = text
    # Banned word detection
    for word in BANNED_WORDS:
        if word.lower() in text.lower():
            logging.warning(f"Banned word detected: {word}")
            text = text.replace(word, "[BANNED_CONTENT]")
    
    # PII detection
    for pattern, replacement in PII_PATTERNS:
        text = re.sub(pattern, replacement, text)
    if original_text != text:
        logging.info(f"Guardrails applied to text: {text}")
    
    return text



def nvdia_guardrails(query: str) -> str:
    """
    Apply NVIDIA NeMo Guardrails to the input query using built-in self-check input rail.
    Returns the original query if it passes the guardrails, None otherwise.
    """
    try:
        import logging
        import sys
        from nemoguardrails import LLMRails, RailsConfig
        from langchain_openai import ChatOpenAI
        
        # Suppress verbose logging
        logging.getLogger("nemoguardrails").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        
        # Load the guardrails configuration
        config = RailsConfig.from_path("./guardrail_config.yml")
        
        # Initialize the guardrails with the LLM
        rails = LLMRails(config)
        
        # Use the built-in self-check input rail
        result = rails.generate(messages=[{
            "role": "user",
            "content": query
        }])

        print(result)
        
        # If the message was blocked, the result will be empty or contain a refusal
        if not result.get("messages"):
            logging.warning(f"Input blocked by guardrails: {query}")
            print("\nI'm sorry, but I can't assist with that request.")
            sys.exit(1)
            
        return query
        
    except Exception as e:
        logging.error(f"Error in NVIDIA Guardrails: {str(e)}", exc_info=True)
        print("\nI'm sorry, but I can't process your request at this time.")
        sys.exit(1)