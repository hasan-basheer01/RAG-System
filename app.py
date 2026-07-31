import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from openai import OpenAI
from langsmith.wrappers import wrap_openai

# -----------------------------
# Load PDF
# -----------------------------
loader = PyPDFLoader("Electric_Circuit_Analysis_Complete_Notes.pdf")
documents = loader.load()

print(f"Loaded {len(documents)} pages")

# -----------------------------
# Split into Chunks
# -----------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

chunks = splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks")

# -----------------------------
# Create Embeddings
# -----------------------------
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# -----------------------------
# Create FAISS Vector Store
# -----------------------------
vector_store = FAISS.from_documents(
    chunks,
    embeddings
)

print("FAISS Index Created")

# -----------------------------
# LangSmith + OpenAI
# -----------------------------
client = wrap_openai(OpenAI())

print("\n==============================")
print("PDF RAG Ready")
print("==============================")

# -----------------------------
# Ask Questions
# -----------------------------
while True:

    question = input("\nAsk a Question (type 'exit' to quit): ")

    if question.lower() == "exit":
        print("Goodbye!")
        break

    # Retrieve Relevant Chunks
    results = vector_store.similarity_search(
        question,
        k=3
    )

    context = "\n\n".join(
        doc.page_content for doc in results
    )

    prompt = f"""
You are a helpful AI assistant.

Answer ONLY using the given context.

If the answer is not available in the context,
say "I couldn't find that information in the document."

Context:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    print("\n==============================")
    print("ANSWER")
    print("==============================")
    print(response.choices[0].message.content)

    print("\n==============================")
    

print("\nOpen https://smith.langchain.com to view traces.")