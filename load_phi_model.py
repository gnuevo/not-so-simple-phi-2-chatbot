import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    StoppingCriteria, 
    StoppingCriteriaList, 
    )
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from typing import List


class StopOnTokens(StoppingCriteria):
    """Stops the model if it produces an 'end of text' token"""
    def __call__(self, input_ids: torch.LongTensor, 
                 scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [50256, 198] # <|endoftext|> and EOL
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False
    

class StopOnNames(StoppingCriteria):
    """
    Stops the model when it starts hallucinating future turns of the 
    conversation.

    It stops the token generation when we find a token sequence of the form 
    "\n<name>:", for example "\nUser:" or "\nAssistant:".
    """
    EOL_TOKEN = 198
    COLON_TOKEN = 25
    
    def __init__(self, tokenized_names: List[List[int]]):
        self.tokenized_names = tokenized_names
    
    def __call__(self, input_ids: torch.LongTensor, 
                 scores: torch.FloatTensor, **kwargs) -> bool:
        for tokens in self.tokenized_names:
            template = [self.EOL_TOKEN, *tokens, self.COLON_TOKEN]
            if input_ids[0][-len(template):].tolist() == template:
                return True
        return False


def load_phi_model_and_tokenizer():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Your device is", device)

    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2", 
        device_map="auto", 
        torch_dtype="auto" if device == "cuda" else torch.float, 
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/phi-2", trust_remote_code=True)
    return model, tokenizer


def get_langchain_model(model, tokenizer):
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200, stopping_criteria=[StopOnTokens()])
    hf = HuggingFacePipeline(pipeline=pipe)
    return hf