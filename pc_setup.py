import time
from pinecone import Pinecone, PodSpec
import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.environ.get('PINECONE_API_KEY')
environment = os.environ.get('PINECONE_ENVIRONMENT')

pc_client = Pinecone(api_key=api_key)
spec = PodSpec(environment=environment)

index_name = "lovecraft"
existing_indexes = [
    index_info["name"] for index_info in pc.list_indexes()
]

# check if index already exists (it shouldn't if this is first time)
if index_name not in existing_indexes:
    # if does not exist, create index
    pc_client.create_index(
        index_name,
        dimension=384,  # dimensionality of minilm
        metric='dotproduct',
        spec=spec
    )
    # wait for index to be initialized
    while not pc_client.describe_index(index_name).status['ready']:
        time.sleep(1)

# connect to index
pc_vs_index = pc_client.Index(index_name)
time.sleep(1)
# view index stats
pc_vs_index.describe_index_stats()