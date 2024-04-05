# RAG-Project-CS-579
A repository for a RAG model project in CS 579 at ISU

## Milestone 1: March 20
We used llamaIndex as the process pipeline and pinecone as vector storage. 

We have a <data_lovecraft> folder which contains a collection of pdf files of H.P. Lovecraft's work that will be loaded into the vector storage.

The text chunk size is set to <512>

The embedding model is set to <bge-small-en-v1.5>

**To Use:** after adding a valid PineCone API key, call `python load.py <your/path/filename.pdf> --index "index_name"` to perform a commandline file upload.
