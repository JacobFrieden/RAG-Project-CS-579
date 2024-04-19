import argparse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pc_setup import pc_client, pc_vs_index
from pathlib import Path

from llama_index.core.retrievers import VectorIndexRetriever
    
def main():
    parser = argparse.ArgumentParser(description="Upload a PDF file to the specified index")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--index", default="default_index", help="Name of the index to upload the PDF to")
    args = parser.parse_args()

    # Initialize settings
    Settings.chunk_size = 512
    Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")

    # Initialize Pinecone vector store
    vector_store = PineconeVectorStore(pinecone_index=pc_client.Index(args.index))
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    document = SimpleDirectoryReader(input_files=[args.pdf_path]).load_data()

    index = VectorStoreIndex.from_documents(
        document, transformations=[SentenceSplitter(chunk_size=512)], storage_context=storage_context
    )

    print("PDF file uploaded successfully!")

if __name__ == "__main__":
    main()
