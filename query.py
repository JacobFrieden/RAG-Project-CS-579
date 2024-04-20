import argparse
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
from llama_index.core.embeddings import resolve_embed_model
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pc_setup import pc_client, pc_vs_index
from llama_index.llms.ollama import Ollama
from llama_index.core.query_engine import RetrieverQueryEngine

def main():
    # Set up argparse to handle command-line arguments
    parser = argparse.ArgumentParser(description="Query a vector index with a specified query.")
    parser.add_argument("--query", type=str, help="The query to retrieve relevant documents.")
    parser.add_argument("--index", type=str, help="The name of the vector index to query.")
    args = parser.parse_args()

    # Initialize settings and settings for the Llama and vector index
    Settings.llm = Ollama(model="llama2", request_timeout=60.0)
    Settings.chunk_size = 512
    Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")

    # Retrieve query and index name from command-line arguments
    query = args.query
    index_name = args.index

    # Initialize vector store and retriever based on the specified index name
    vector_store = PineconeVectorStore(pinecone_index=pc_client.Index(index_name))
    vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=3)

    # Initialize query engine for retrieval
    query_engine_retrieval = RetrieverQueryEngine(retriever=retriever)

    # Perform query retrieval and print results
    print(query_engine_retrieval.query(query))

if __name__ == "__main__":
    main()