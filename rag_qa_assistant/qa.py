import os
import argparse
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA

# Load environment variables from .env file (for API keys etc)
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


# 1. Load and split documents into manageable chunks

def load_and_split_docs(directory):
    all_docs = []
    print(f"Attempting to load documents from: {directory}")

    if not os.path.exists(directory):
        print(f"Error: Document directory '{directory}' does not exist.")
        return []  # No docs to process

    found_files = False
    for filename in os.listdir(directory):
        # Only consider .txt and .md files
        if filename.endswith(".txt") or filename.endswith(".md"):
            found_files = True
            file_path = os.path.join(directory, filename)
            print(f"Processing file: {file_path}")
            try:
                loader = TextLoader(file_path, encoding="utf-8")
                documents = loader.load()
                print(f"  Loaded {len(documents)} document(s) from {filename}")

                if not documents:
                    print(f"  Warning: File '{filename}' contained no loadable content. Skipping.")
                    continue  # Skip empty files

                # Break docs into overlapping chunks for better search/QA
                splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                chunks = splitter.split_documents(documents)
                print(f"  Split {len(documents)} document(s) from '{filename}' into {len(chunks)} chunks.")

                if not chunks:
                    print(
                        f"  Warning: Splitting '{filename}' resulted in 0 chunks. Chunk size/overlap might be too restrictive or content too small.")

                all_docs.extend(chunks)
            except Exception as e:
                print(f"  Error processing file '{filename}': {e}. Skipping this file.")
                # Don't let one bad file stop the process

    if not found_files:
        print(f"No .txt or .md files found in '{directory}'.")

    print(f"Total documents loaded and split into chunks: {len(all_docs)}")
    return all_docs


# 2. Create a new vectorstore, or load an existing one if possible

def create_or_load_vectorstore(documents, persist_path="./vector_store"):
    # Make sure the vector_store directory exists
    os.makedirs(persist_path, exist_ok=True)

    # If there's already a saved index, use it (much faster)
    if os.path.exists(persist_path) and os.listdir(persist_path):
        print(f"Loading existing vectorstore from {persist_path}...")
        return FAISS.load_local(
            persist_path,
            OpenAIEmbeddings(),
            allow_dangerous_deserialization=True  # Needed for safety in newer FAISS
        )
    else:
        print(f"Creating new vectorstore at {persist_path}...")
        if not documents:
            raise ValueError("Cannot create a vectorstore from an empty list of documents.")
        vectorstore = FAISS.from_documents(documents, OpenAIEmbeddings())
        vectorstore.save_local(persist_path)
        print("Vectorstore created and saved.")
        return vectorstore

# 3. Generate an answer for a question, using your documents

def answer_question(question, docs_path="./docs"):
    documents = load_and_split_docs(docs_path)

    if not documents:
        raise ValueError(
            "No documents were loaded or processed successfully. Please check your 'docs' directory and its contents.")

    vectorstore = create_or_load_vectorstore(documents)
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain.run(question)

# -----
# 4. Command-line interface
# -----
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLI QA Assistant using LangChain + OpenAI")
    parser.add_argument("--question", type=str, required=True, help="Your question")
    parser.add_argument("--docs", type=str, default="./docs", help="Path to documents directory")
    args = parser.parse_args()

    try:
        print("QA Assistant Started !")
        print(f"Question: {args.question}")
        print(f"Documents Path: {args.docs}")
        answer = answer_question(args.question, args.docs)
        print("\nAnswer")
        print(answer)
        print("QA Assistant Finished-")
    except Exception as e:
        print("Error details:", str(e))
        print("Please check your documents, API key, and installed dependencies.")