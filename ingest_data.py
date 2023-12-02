from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings

loader = PyPDFLoader("data/contract.pdf")
pages = loader.load_and_split()

from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(pages)

import weaviate
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Weaviate
from weaviate.embedded import EmbeddedOptions

client = weaviate.Client(embedded_options=EmbeddedOptions())

vectorstore = Weaviate.from_documents(
    client=client, documents=chunks, embedding=OpenAIEmbeddings(), by_text=False
)
