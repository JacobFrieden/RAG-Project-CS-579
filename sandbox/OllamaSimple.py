from llama_index.llms.ollama import Ollama
from llama_index.core import Settings

Settings.llm = Ollama(model="llama2", request_timeout=60.0)

response = Settings.llm.complete("Paul Graham is ")
print(response)
