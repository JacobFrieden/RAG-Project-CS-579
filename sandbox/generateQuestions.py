from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
import os
import nest_asyncio
nest_asyncio.apply()

import openai
openai.api_key = os.environ.get('OPENAI_API_KEY')

# wikipedia pages
from llama_index.readers.wikipedia import WikipediaReader
from llama_index.core import VectorStoreIndex

cities = [
    "San Francisco",
]

documents = WikipediaReader().load_data(
    pages=[f"History of {x}" for x in cities]
)
index = VectorStoreIndex.from_documents(documents)

# documents = SimpleDirectoryReader("./data_small/").load_data()

# generate questions against chunks
from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama

# set context for llm provider
# llm_openai = OpenAI(model="gpt-3.5-turbo", temperature=0.3)
llm_ollama = Ollama(model= "llama2:13b", verbose=True)

# instantiate a DatasetGenerator
dataset_generator = RagDatasetGenerator.from_documents(
    documents[0:2],
    llm=llm_ollama,
    num_questions_per_chunk=2,  # set the number of questions per nodes
    show_progress=True,
)

rag_dataset = dataset_generator.generate_dataset_from_nodes()
rag_dataset.to_pandas()
rag_dataset.save_json("rag_dataset.json")


