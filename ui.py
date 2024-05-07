import funix

# Import the necessary modules
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.embeddings import resolve_embed_model
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pc_setup import pc_client, pc_vs_index
from llama_index.llms.ollama import Ollama
from llama_index.core.query_engine import RetrieverQueryEngine

# Set up the Llama and vector index settings
Settings.llm = Ollama(model="llama2", request_timeout=60.0)
Settings.chunk_size = 512
Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")

def query_vector_index(query: str, index_name: str) -> str:
    """Query the vector index with a specified query."""
    # Initialize the vector store and retriever
    vector_store = PineconeVectorStore(
        pinecone_index=pc_client.Index(index_name))
    vector_index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store)
    retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=3)

    # Initialize the query engine and perform the query
    query_engine_retrieval = RetrieverQueryEngine(retriever=retriever)
    results = query_engine_retrieval.query(query)

    return str(results)

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pc_setup import pc_client, pc_vs_index
from llama_index.core.retrievers import VectorIndexRetriever

# @funix.funix(description="Select a pdf to upload")
def pdf_upload_box(file_path: str, index_name: str = "lovecraft") -> str:
    Settings.chunk_size = 512
    Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")
    print(file_path)
    
    vector_store = PineconeVectorStore(pinecone_index=pc_client.Index(index_name))
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    document = SimpleDirectoryReader(input_files=[file_path]).load_data()

    index = VectorStoreIndex.from_documents(
        document, transformations=[SentenceSplitter(chunk_size=512)], storage_context=storage_context
    )

    return str(f"PDF file {file_path} uploaded successfully!")
