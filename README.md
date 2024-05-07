# RAG-Project-CS-579
A repository for a RAG model project in CS 579 at ISU

## Milestone 1: March 20
We used llamaIndex as the process pipeline and pinecone as vector storage. 

We have a <data_lovecraft> folder which contains a collection of pdf files of H.P. Lovecraft's work that will be loaded into the vector storage.

The text chunk size is set to <512>

The embedding model is set to <bge-small-en-v1.5>

**To Use:** after adding a valid PineCone API key to the source code, call `python load.py <your/path/filename.pdf> --index "index_name"` to perform a commandline file upload.

https://github.com/JacobFrieden/RAG-Project-CS-579/assets/116682778/7581c1db-baee-4452-a1e2-bedc366c5a57

## Milestone 2: April 19
We used Ollama's llama2 model as the llm backbone that handles the answering.

We used *VectorIndexRetriever* to retrive top-k similar chunk that's relevent to the query and generated the answer using *RetrieverQueryEngine* based on the retrived information.

**To Use:** after adding a valid PineCone API key to the source code, call `python query.py --query "What do you want to know?" --index "index_name"` to perform an answering generation based on the files contained in the pinecone vector store index.

Additionally, the user will need to either be running a local instance of Ollama with a llama2 model loaded, or otherwise take steps to modify the LLM used by the `llama_index` library for our query engine to run. 

https://github.com/JacobFrieden/RAG-Project-CS-579/assets/45891316/3cb1b396-d7ed-4992-b956-ccc83a6d26d6

## Milestone 3: May 7th
We used funix from https://github.com/TexteaInc/funix as the ui library for this project. To test, you have to intsall Ollama locally and install funix via pip.

https://github.com/JacobFrieden/RAG-Project-CS-579/assets/45891316/ef7642f5-4149-4ad8-b1d2-a25851f834a7




