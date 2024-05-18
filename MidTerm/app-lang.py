import os
from markdownify import markdownify as md
from dotenv import load_dotenv
import chainlit as cl
from langchain_community.vectorstores import Qdrant
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import pickle
from qdrant_client import QdrantClient
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
import pdfplumber
import tqdm

# Load environment variables
load_dotenv()

# Set OpenAI model and embedding models
openai_chat_model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
embed_model = OpenAIEmbeddings(model="text-embedding-3-small")

# Function to create text embeddings with progress tracking
def create_text_embeddings(chunks, embedding_model):
    embeddings = []
    for chunk in tqdm.tqdm(chunks, desc="Creating Text Embeddings"):
        try:
            embedding = embedding_model.embed_query(chunk)
            embeddings.append({'text': chunk, 'embedding': embedding})
        except Exception as e:
            print(f"Failed to create embedding for chunk: {chunk}, error: {e}")
    return embeddings

# Function to convert PDF to Markdown
def convert_pdf_to_md(pdf_path, md_path):
    with pdfplumber.open(pdf_path) as pdf:
        pdf_text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        md_content = md(pdf_text)
        with open(md_path, 'w', encoding='utf-8') as md_file:
            md_file.write(md_content)
    return md_content

# Function to process PDF file and chunk text
def process_and_chunk_pdf(pdf_file, chunker):
    chunks = []
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                page_chunks = chunker.split_text(text)
                chunks.extend(page_chunks)
    return chunks

# Example usage
pdf_path = 'DataRepository/Meta10K.pdf'
output_dir = 'DataRepository'
md_filename = os.path.splitext(os.path.basename(pdf_path))[0] + '.md'
md_path = os.path.join(output_dir, md_filename)
convert_pdf_to_md(pdf_path, md_path)

# Initialize SemanticChunker
chunker = SemanticChunker(embed_model,breakpoint_threshold_type="percentile")  # Initialize without chunk_size

# Process PDF and chunk text
chunks = process_and_chunk_pdf(pdf_path, chunker)

# File path to store embeddings
text_embedding_file = "text_embeddings.pkl"

# Create or load text embeddings from file
if os.path.exists(text_embedding_file):
    with open(text_embedding_file, 'rb') as f:
        text_embeddings = pickle.load(f)
else:
    text_embeddings = create_text_embeddings(chunks, embed_model)
    with open(text_embedding_file, 'wb') as f:
        pickle.dump(text_embeddings, f)

# Define a simple Document class
class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}

# Convert embeddings to Document objects
documents = [Document(item['text'], {'embedding': item['embedding']}) for item in text_embeddings]

# Create QDrant Vector store index
qdrant_vectorstore = Qdrant.from_documents(
    documents,
    embed_model,  # Correct usage of embedding method
    location=":memory:",
    collection_name="Meta_10K",
)

qdrant_retriever = qdrant_vectorstore.as_retriever()

# Load the QA chain
combine_documents_chain = load_qa_chain(llm=openai_chat_model, chain_type="stuff")

# Initialize Qdrant Client
client = QdrantClient(location=":memory:")

qa_chain = RetrievalQA(
    retriever=qdrant_retriever,
    combine_documents_chain=combine_documents_chain,
)

@cl.on_chat_start
async def start_chat():
    settings = {
        "model": "gpt-3.5-turbo",
        "temperature": 0,
        "max_tokens": 500,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }
    cl.user_session.set("settings", settings)

@cl.on_message
async def main(message: cl.Message):
    query = message.content
    response = qa_chain.run(query)
    msg = cl.Message(content=response)
    await msg.send()
