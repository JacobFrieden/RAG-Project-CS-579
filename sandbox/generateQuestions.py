from llama_index.readers.web import BeautifulSoupWebReader

url = "https://www.theverge.com/2023/9/29/23895675/ai-bot-social-network-openai-meta-chatbots"

documents = BeautifulSoupWebReader().load_data([url])

import torch
from transformers import BitsAndBytesConfig
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.huggingface import HuggingFaceLLM

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

def messages_to_prompt(messages):
  prompt = ""
  for message in messages:
    if message.role == 'system':
      prompt += f"<|system|>\n{message.content}</s>\n"
    elif message.role == 'user':
      prompt += f"<|user|>\n{message.content}</s>\n"
    elif message.role == 'assistant':
      prompt += f"<|assistant|>\n{message.content}</s>\n"

  # ensure we start with a system prompt, insert blank if needed
  if not prompt.startswith("<|system|>\n"):
    prompt = "<|system|>\n</s>\n" + prompt

  # add final assistant prompt
  prompt = prompt + "<|assistant|>\n"

  return prompt


llm = HuggingFaceLLM(
    model_name="HuggingFaceH4/zephyr-7b-alpha",
    tokenizer_name="HuggingFaceH4/zephyr-7b-alpha",
    query_wrapper_prompt=PromptTemplate("<|system|>\n</s>\n<|user|>\n{query_str}</s>\n<|assistant|>\n"),
    context_window=3900,
    max_new_tokens=256,
    model_kwargs={"quantization_config": quantization_config},
    # tokenizer_kwargs={},
    generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95},
    messages_to_prompt=messages_to_prompt,
    device_map="auto",
)

from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
Settings.llm = llm
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

from llama_index.core import VectorStoreIndex
vector_index = VectorStoreIndex.from_documents(documents)

from llama_index.core import SummaryIndex
summary_index = SummaryIndex.from_documents(documents)

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

query_engine = vector_index.as_query_engine(response_mode="compact")
response = query_engine.query("How do OpenAI and Meta differ on AI tools?")
print(response)

query_engine = vector_index.as_query_engine(response_mode="refine")
response = query_engine.query("How do OpenAI and Meta differ on AI tools?")
print(response)



