import argparse
from pathlib import Path
from pprint import pprint
from typing import List, Tuple
from typing import Literal

from langchain import hub
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph import END, StateGraph, START
from pydantic import BaseModel
from typing_extensions import TypedDict
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError


def with_retry(runnable, max_retries=3):
    """Wraps a Runnable with retry logic for structured output validation."""
    def invoke_with_retry(input, config=None, **kwargs):
        attempts = 0
        last_exception = None
        while attempts < max_retries:
            try:
                return runnable.invoke(input, config=config, **kwargs)
            except ValidationError as e:
                last_exception = e
                attempts += 1
                print(f"Validation error on attempt {attempts}, retrying...")
            except Exception as e:
                last_exception = e
                attempts += 1
                print(f"Error on attempt {attempts}: {e}, retrying...")
        raise Exception(f"Failed after {max_retries} attempts. Last error: {last_exception}") from last_exception
    return RunnableLambda(invoke_with_retry)

class SubQuery(BaseModel):
    """Subquestion decomposition model"""
    questions: List[str] = Field(description="List of sub questions")

class GradeDocuments(BaseModel):
    """Document relevance grading model"""
    yes_or_no: Literal['yes', 'no'] = Field(description="Relevance score 'yes' or 'no'")

class GradeHallucinations(BaseModel):
    """Hallucination grading model"""
    yes_or_no: Literal['yes', 'no'] = Field(description="Grounded in facts 'yes' or 'no'")

class GradeAnswer(BaseModel):
    """Answer quality grading model"""
    yes_or_no: Literal['yes', 'no'] = Field(description="Addresses question 'yes' or 'no'")


class GraphState(TypedDict):
    """State management for processing graph"""
    question: str
    sub_questions: List[str]
    generation: str
    documents: List[str]


def parse_arguments():
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


def initialize_components(args):
    """Initialize system components"""
    # Document loading and processing
    loader = DirectoryLoader(
        str(args.input_folder),
        glob="**/*.txt",
        loader_cls=TextLoader
    )

    docs = []
    for doc in loader.load():
        source = str(doc.metadata["source"]).replace(str(args.input_folder), "").lstrip('/')
        if '[' in source:
            title_rest, source_part = source.split("[")[:-1], source.split("[")[-1]
            video_title = "[".join(title_rest)
            video_id = source_part.split("]")[0]
            doc.metadata["video_id"] = video_id
            doc.metadata["video_title"] = video_title
        docs.append(doc)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    doc_splits = text_splitter.split_documents(docs)

    # Embeddings and retrievers
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    reranker = FlashrankRerank()

    bm25_retriever = BM25Retriever.from_documents(doc_splits, k=args.top_k)
    faiss_vectorstore = FAISS.from_documents(doc_splits, embeddings)
    faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": args.top_k})

    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever], 
        weights=[0.7, 0.3]
    )

    # LLM configuration
    llm = ChatOllama(
        model="llama3.1:8b",
        temperature=0.8,
        num_predict=1024,
    )

    return reranker, hybrid_retriever, llm


def build_workflow(reranker, hybrid_retriever, llm):
    """Construct and configure the processing workflow"""
    # Initialize components
    sub_question_generator = with_retry(llm.with_structured_output(SubQuery), max_retries=5)
    prompt = hub.pull("rlm/rag-prompt")
    rag_chain = prompt | llm | StrOutputParser()

    # Grading prompts
    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", """You assess document relevance to questions. If document contains
         related keywords/semantics, answer. Respond only with 'yes' or 'no'."""),
        ("human", "Document:\n\n{document}\n\nQuestion: {question}")
    ])
    retrieval_grader = grade_prompt | with_retry(llm.with_structured_output(GradeDocuments), max_retries=5)

    hallucination_prompt = ChatPromptTemplate.from_messages([
        ("system", "Determine if the answer is grounded in facts. Respond only with 'yes' or 'no'."),
        ("human", "Facts:\n\n{documents}\n\nAnswer: {generation}")
    ])
    hallucination_grader = hallucination_prompt | with_retry(llm.with_structured_output(GradeHallucinations), max_retries=5)

    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", "Does answer resolve the question? Respond only with 'yes' or 'no'."),
        ("human", "Question:\n\n{question}\n\nAnswer: {generation}")
    ])
    answer_grader = answer_prompt | with_retry(llm.with_structured_output(GradeAnswer), max_retries=5)

    question_rewriter = ChatPromptTemplate.from_messages([
        ("system", "Improve this question for better search results."),
        ("human", "Original question: {question}")
    ]) | llm | StrOutputParser()

    # Workflow construction
    workflow = StateGraph(GraphState)

    # Node definitions
    workflow.add_node("decompose", lambda state: {
        "sub_questions": sub_question_generator.invoke(state["question"]).questions,
        "question": state["question"]
    })
    workflow.add_node("retrieve", lambda state: {
        "documents": [doc for q in state["sub_questions"] 
                     for doc in hybrid_retriever.get_relevant_documents(q)],
        "question": state["question"]
    })
    workflow.add_node("rerank", lambda state: {
        "documents": reranker.compress_documents(
            query=state["question"], 
            documents=state["documents"]
        ),
        "question": state["question"]
    })
    workflow.add_node("grade_documents", lambda state: {
        "documents": [doc for doc in state["documents"] 
                     if retrieval_grader.invoke({
                         "question": state["question"],
                         "document": doc.page_content
                     }).yes_or_no == "yes"],
        "question": state["question"]
    })
    workflow.add_node("generate", lambda state: {
        "generation": rag_chain.invoke({
            "context": state["documents"],
            "question": state["question"]
        }),
        "documents": state["documents"],
        "question": state["question"]
    })
    workflow.add_node("transform_query", lambda state: {
        "question": question_rewriter.invoke({"question": state["question"]}),
        "documents": state["documents"]
    })

    # Graph connections
    workflow.add_edge(START, "decompose")
    workflow.add_edge("decompose", "retrieve")
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "grade_documents")
    
    workflow.add_conditional_edges(
        "grade_documents",
        lambda state: "transform_query" if not state["documents"] else "generate",
        {"transform_query": "transform_query", "generate": "generate"}
    )
    
    workflow.add_conditional_edges(
        "generate",
        lambda state: (
            "transform_query" if hallucination_grader.invoke({
                "documents": state["documents"],
                "generation": state["generation"]
            }).yes_or_no == "no" 
            else END if answer_grader.invoke({
                "question": state["question"],
                "generation": state["generation"]
            }).yes_or_no == "yes" 
            else "transform_query"
        ),
        {"transform_query": "transform_query", END: END}
    )
    
    workflow.add_edge("transform_query", "retrieve")

    return workflow.compile()


def main():
    args = parse_arguments()
    
    reranker, hybrid_retriever, llm = initialize_components(args)
    workflow = build_workflow(reranker, hybrid_retriever, llm)
    
    print(f"\n{'='*40}\nProcessing question: {args.question}\n{'='*40}")
    
    result = None
    for output in workflow.stream(
        {"question": args.question},
        {"recursion_limit": args.max_recursion}
    ):
        for key, value in output.items():
            pprint(f"Node '{key}':")
            print(value)
        result = output.get(list(output.keys())[-1])
        pprint("\n---\n")

    if result and "generation" in result:
        pprint("Final Answer:")
        print(result["generation"])
        
        unique_sources = {
            (doc.metadata.get("video_title", "Unknown"), 
             doc.metadata.get("video_id", "Unknown"))
            for doc in result["documents"]
        }
        
        print("\nReference Sources:")
        for title, vid in unique_sources:
            print(f"- {title} (https://www.youtube.com/watch?v={vid})")


if __name__ == "__main__":
    main()