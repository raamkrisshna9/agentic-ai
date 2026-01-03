#################
# postprocess.py
#################
# This is place to add post processing steps,
# - Trustworthiness checks
# - Citation Injection / Context Reference / File Location
# - Grounding check - check if the response is grounded in the context
# - Data Normalization - format the response to the user's expected format like JSON
# - Addition of metadata like Confidence score, latency, used docs
# - Removal of values of certain formats like SSN, Phone numbers, etc.

import re

# Remove PII from the output
PII_REGEX = re.compile(r'\b\d{3}-\d{2}-\d{4}\b') # Social Security Number
def secured_output(response: str) -> str:
    if PII_REGEX.search(response):
        return "[REDACTED]"
    return response.strip()