from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pc_setup import pc_client, pc_vs_index
Settings.chunk_size = 512
vector_store = PineconeVectorStore(pinecone_index=pc_vs_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

documents = SimpleDirectoryReader("data_lovecraft").load_data()

# bge embedding model
Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")

index = VectorStoreIndex.from_documents(
    documents, transformations=[SentenceSplitter(chunk_size=512)], storage_context=storage_context
)
