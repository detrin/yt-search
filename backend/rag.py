import argparse
from pathlib import Path

# Document loading & retrieval
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

# LLM
from langchain_ollama import ChatOllama

# Prompts
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Ensemble retriever
from langchain.retrievers import EnsembleRetriever


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Document QA System")
    parser.add_argument("--input_folder", type=Path, required=True,
                        help="Path to directory containing text documents")
    parser.add_argument("--question", type=str, required=True,
                        help="Question to ask about the documents")
    parser.add_argument("--top_k", type=int, default=3,
                        help="Number of documents to retrieve (default: 3)")
    parser.add_argument("--max_recursion", type=int, default=50,
                        help="Maximum recursion depth (default: 50)")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1) Load documents
    loader = DirectoryLoader(
        str(args.input_folder),
        glob="**/*.txt",
        loader_cls=TextLoader
    )
    docs = []
    for doc in loader.load():
        # Extract video info from filename if present
        source = str(doc.metadata["source"]).replace(str(args.input_folder), "").lstrip('/')
        if '[' in source:
            title_rest, source_part = source.split("[")[:-1], source.split("[")[-1]
            video_title = "[".join(title_rest)
            video_id = source_part.split("]")[0]
            doc.metadata["video_id"] = video_id
            doc.metadata["video_title"] = video_title
        else:
            doc.metadata["video_id"] = "Unknown"
            doc.metadata["video_title"] = doc.metadata["source"]
        docs.append(doc)

    # 2) Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    doc_chunks = splitter.split_documents(docs)

    # 3) Build retriever (hybrid of BM25 + FAISS)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    bm25_retriever = BM25Retriever.from_documents(doc_chunks, k=args.top_k)
    vectorstore = FAISS.from_documents(doc_chunks, embeddings)
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": args.top_k})

    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.7, 0.3]
    )

    # 4) Fetch the most relevant chunks
    relevant_docs = hybrid_retriever.get_relevant_documents(args.question)

    # 5) Create a simple prompt & LLM
    #    For demonstration, we'll pull a prompt from the hub or create our own.
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer the user based on the provided context."),
        ("human", "Context:\n\n{context}\n\nQuestion: {question}")
    ])
    llm = ChatOllama(
        model="llama3.1:8b",
        temperature=0.8,
        num_predict=1024,
        base_url='http://ollama:11434',
    )
    # Convert to chain
    chain = rag_prompt | llm | StrOutputParser()

    # 6) Prepare context text and get LLM answer
    context_text = "\n\n".join([doc.page_content for doc in relevant_docs])
    answer = chain.invoke({
        "context": context_text,
        "question": args.question
    }, {"recursion_limit": args.max_recursion})

    # 7) Print final answer
    # print("="*40)
    print("Question:", args.question)
    print("Answer:", answer)

    # 8) Print reference video sources
    unique_sources = {
        (doc.metadata.get("video_title", "Unknown"),
         doc.metadata.get("video_id", "Unknown"))
        for doc in relevant_docs
    }
    print("Reference YouTbe videos:")
    for title, vid in unique_sources:
        print(f"- [{title}](https://www.youtube.com/watch?v={vid})")

if __name__ == "__main__":
    main()