import torch
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
from llama_index.core.embeddings import resolve_embed_model
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pc_setup import pc_client, pc_vs_index
from transformers import BitsAndBytesConfig
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.ollama import Ollama

Settings.llm = Ollama(model="llama2", request_timeout=60.0)
Settings.chunk_size = 512
Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")

# vector_store = PineconeVectorStore(pinecone_index=pc_client.Index("lovecraft"))
# vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
# retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=3)
# retrived_texts = retriever.retrieve('Who worshipped the Old Ones?')
# print(retrived_texts[0])

import logging
import sys
import pandas as pd

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index.readers.wikipedia import WikipediaReader
from llama_index.core import VectorStoreIndex

cities = [
    "San Francisco",
]

documents = WikipediaReader().load_data(
    pages=[f"History of {x}" for x in cities]
)

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Response
from llama_index.core.llama_dataset.generator import RagDatasetGenerator

# reader = SimpleDirectoryReader("./data_small/")

# documents = reader.load_data(
#     show_progress=True
# )

dataset_generator = RagDatasetGenerator.from_documents(
    documents=documents,
    llm=Settings.llm,
    num_questions_per_chunk=2,
    show_progress=True,
)

print(len(dataset_generator.nodes))
rag_dataset = dataset_generator.generate_questions_from_nodes()
rag_dataset.to_pandas()
rag_dataset.save_json("rag_dataset.json")

from llama_index.core.query_engine import RetrieverQueryEngine

# query_engine = RetrieverQueryEngine(retriever=retriever)

# llm_query = query_engine.query('Who worshipped the Old Ones?')

# print(llm_query)


