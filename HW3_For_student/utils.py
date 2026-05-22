# from typing import List
# import re
    
# def get_inference_system_prompt() -> str:
#     """get system prompt for generation"""
#     prompt = "" 
#     return prompt

# def get_inference_user_prompt(query : str, context_list : List[str]) -> str:
#     """Create the user prompt for generation given a query and a list of context passages."""
#     prompt = f""""""
#     return prompt

# def parse_generated_answer(pred_ans: str) -> str:
#     """Extract the actual answer from the model's generated text."""
#     parsed_ans = pred_ans
#     return parsed_ans

from typing import List
import re

def get_inference_system_prompt() -> str:
    """get system prompt for generation"""
    prompt = "You are a helpful assistant that extracts the exact answer span from provided passages. If no relevant answer is found, respond with only: CANNOTANSWER."
    return prompt

def get_inference_user_prompt(query: str, context_list: List[str]) -> str:
    """Create the user prompt for generation given a query and a list of context passages."""
    context = "\n\n".join(context_list)
    prompt = f"""Extract the exact answer span from the passages below that answers the question. 
Only copy the span directly from the passages without any additional explanation or modification. 
If none of the passages contain the answer, respond with only: CANNOTANSWER.

Passages:
{context}

Question:
{query}

Answer:"""
    return prompt

def parse_generated_answer(pred_ans: str) -> str:
    """Extract the actual answer from the model's generated text."""
    parsed_ans = pred_ans.strip()
    return parsed_ans
