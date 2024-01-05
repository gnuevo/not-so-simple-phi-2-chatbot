import gradio as gr
import torch
from load_phi_model import StopOnTokens, StopOnNames
from transformers import StoppingCriteriaList, TextIteratorStreamer
from threading import Thread
import re

from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda

from load_phi_model import load_phi_model_and_tokenizer, get_langchain_model
model, tokenizer = load_phi_model_and_tokenizer()
hf = get_langchain_model(model, tokenizer)

classification_template = """
Instruct: Classify the following text in one of the following categories: \
["support", "sales", "joke"]. Output only the name of the category.
+ "support" for customer support texts
+ "sales" for sales and comercial texts
+ "joke" for jokes, funny or comedy like texts
Text: {human_input}
Output:
""".strip()
classification_prompt = ChatPromptTemplate.from_template(classification_template)
classification_chain = (
    classification_prompt
    | hf
    | StrOutputParser()
)


support_instructions = """\
You are a customer support agent. It seems that the user may have some \
issues. Answer to their query politely and sincerely. Be kind, understanding \
and say you're sorry for the inconvenience or the situation whenever \
necessary. Be brief and to the point."""
sales_instructions = """\
You are an aggressive salesperson. The user is looking for some information \
on products. Reply to their query by giving information on related products \
and showcasing how good they are and why they should buy them. Be brief and \
to the point."""
joke_instructions = """\
You are a comedian. The user want's to have some fun. \
Reply to their query in a funny way."""
general_instructions = """\
Instruction: Respond to the following query."""
support_chain = PromptTemplate.from_template(support_instructions)
sales_chain = PromptTemplate.from_template(sales_instructions)
joke_chain = PromptTemplate.from_template(joke_instructions)
general_chain = PromptTemplate.from_template(general_instructions)

branch = RunnableBranch(
    (lambda x: "support" in x["topic"].lower(), support_chain),
    (lambda x: "sales" in x["topic"].lower(), sales_chain),
    (lambda x: "joke" in x["topic"].lower(), joke_chain),
    general_chain,
) | RunnableLambda(lambda x: x.text)

branch_chain = {
        "topic": classification_chain, 
        "human_input": lambda x: x["human_input"]
    } | branch


template = """\
You are a chatbot having a conversation with a human. Follow the given \
instructions to reply to the Human message below.

Instructions:{instructions}

{chat_history}
Human: {human_input}
Chatbot:"""
prompt = PromptTemplate(
    input_variables=["instructions", "chat_history", "human_input"], 
    template=template
)
chat_chain = (
    {
        "human_input": lambda x: x["human_input"], 
        "instructions": lambda x: branch_chain,
        "chat_history": lambda x: x["chat_history"],
    } | prompt
)


HUMAN_NAME = "Human"
BOT_NAME = "Chatbot"
DEBUG = False # set to True to see the prompt sent to the model


device = "cuda" if torch.cuda.is_available() else "cpu"
chat_name_pattern_end = r'\n.+:$' # matches substrings like `\nUser:` at the end

def predict(message, history):
    stop_on_tokens = StopOnTokens()
    stop_on_names = StopOnNames(
        [tokenizer.encode(HUMAN_NAME), tokenizer.encode(BOT_NAME)])

    messages = "".join(["".join(
        [f"\n{HUMAN_NAME}: "+item[0], f"\n{BOT_NAME}:"+item[1]]
    ) for item in history]).strip()

    input_dict = {
        "human_input": message,
        "chat_history": messages,
    }

    input_prompt = chat_chain.invoke(input_dict).text
    if DEBUG: print(input_prompt)

    model_inputs = tokenizer([input_prompt], return_tensors="pt").to(device)
    streamer = TextIteratorStreamer(tokenizer, timeout=10., 
                                    skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        model_inputs,
        streamer=streamer,
        max_new_tokens=256,
        do_sample=True,
        top_p=0.95,
        top_k=1000,
        temperature=1.0,
        num_beams=1,
        stopping_criteria=StoppingCriteriaList([stop_on_tokens, stop_on_names])
        )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    partial_message = ""
    for new_token in streamer:
        partial_message += new_token
        match = re.search(chat_name_pattern_end, partial_message)
        if match:
            partial_message = partial_message[:-len(match.group())]
        yield partial_message
        

gr.ChatInterface(predict).queue().launch(server_name="0.0.0.0")