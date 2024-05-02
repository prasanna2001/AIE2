# You can find this code for Chainlit python streaming here (https://docs.chainlit.io/concepts/streaming/python)

# OpenAI Chat completion
import chainlit as cl  # importing chainlit for our app
import llama_index

llama_index.version = "0.10.33"

from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    SimpleDirectoryReader,
    load_index_from_storage,
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.core.callbacks import CallbackManager
from llama_index.core.service_context import ServiceContext
from llama_index.core.node_parser import (
    SemanticSplitterNodeParser,
)
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

from dotenv import load_dotenv




###Set our LLM and Embedding model
Settings.llm = OpenAI(model="gpt-3.5-turbo",temperature=0, max_tokens=500, 
                      streaming=True, top_p=1, frequency_penalty=0, presence_penalty=0 )
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

load_dotenv()

#set_global_handler("wandb", run_args={"project": "llama-index-rag"})
#wandb_callback = llama_index.core.global_handler

##Prepare Doc to be parsed. We will convert the pdf to markdown since 
##the document contains tables in images
import PyPDF2
from markdownify import markdownify as md

def pdf_to_text(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        num_pages = len(pdf_reader.pages)
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text

def convert_to_markdown(pdf_path, markdown_path):
    text = pdf_to_text(pdf_path)
    markdown_text = md(text)
    with open(markdown_path, 'w', encoding='utf-8') as md_file:
        md_file.write(markdown_text)

# Example usage:
pdf_path = 'DataRepository/Meta10K.pdf'
markdown_path = 'DataRepository/Meta10K.md'
convert_to_markdown(pdf_path, markdown_path)
print(f"PDF converted to Markdown. Markdown file saved as '{markdown_path}'.")

###Create nodes (doc chunks)
reader = SimpleDirectoryReader(input_dir="DataRepository", exclude=["DataRepository/Meta10K.pdf"])
documents = reader.load_data()


splitter = SemanticSplitterNodeParser(
    buffer_size=1, breakpoint_percentile_threshold=95, embed_model=Settings.embed_model
)

nodes = splitter.get_nodes_from_documents(documents)

###Create QDrant Vector store index
client = QdrantClient(location=":memory:")

client.create_collection(
    collection_name="Meta_10K",
    vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
)

###create our `VectorStore` and `StorageContext` which will allow us to create an empty `VectorStoreIndex` which we will be able to add nodes to later!
vector_store = QdrantVectorStore(client=client, collection_name="Meta_10K")

try:
    # rebuild storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    # load index
    index = load_index_from_storage(storage_context)
except:
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist()

### loop through our nodes, index and construct nodes.
##for node in nodes:
##    index.insert_nodes(nodes)

### Persisting and Loading Stored Index with Weights and Biases
##wandb_callback.persist_index(index, index_name="meta10K-index-qdrant")
##storage_context = wandb_callback.load_storage_context(
##    artifact_url="om-ai/llama-index-meta-rag/meta10K-index-qdrant:v0"
##)


@cl.on_chat_start  # marks a function that will be executed at the start of a user session
async def start_chat():
    service_context = ServiceContext.from_defaults(callback_manager=CallbackManager([cl.LlamaIndexCallbackHandler()]))
    query_engine = index.as_query_engine(streaming=True, similarity_top_k=2, service_context=service_context)
    cl.user_session.set("query_engine", query_engine)
    await cl.Message(
        author="Assistant", content="Hello! Im an AI assistant. How may I help you?"
    ).send()

    ##cl.user_session.set("settings", settings)


@cl.on_message  # marks a function that should be run each time the chatbot receives a message from a user
async def main(message: cl.Message):
    query_engine = cl.user_session.get("query_engine") # type: RetrieverQueryEngine

    msg = cl.Message(content="", author="Assistant")

    res = await cl.make_async(query_engine.query)(message.content)

    for token in res.response_gen:
        await msg.stream_token(token)
    await msg.send()

