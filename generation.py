import torch
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
from llama_index.core.embeddings import resolve_embed_model
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pc_setup import pc_client, pc_vs_index
from llama_index.llms.ollama import Ollama
from llama_index.core.query_engine import RetrieverQueryEngine


Settings.llm = Ollama(model="llama2", request_timeout=60.0)
Settings.chunk_size = 512
Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")

query = "Who worshipped Dagon?"

vector_store = PineconeVectorStore(pinecone_index=pc_client.Index("lovecraft"))
vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=3)
retrived_texts = retriever.retrieve(query)
print(retrived_texts[0])

query_engine = RetrieverQueryEngine(retriever=retriever)
llm_query = query_engine.query(query)
print(llm_query)


