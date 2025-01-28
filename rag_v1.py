from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain.retrievers.document_compressors import FlashrankRerank

from langchain_ollama import ChatOllama
from pydantic import BaseModel

from typing import Literal, Optional, Tuple, List
from langchain_core.pydantic_v1 import BaseModel, Field

from langchain import hub
from langchain_core.output_parsers import StrOutputParser

from langchain_core.prompts import ChatPromptTemplate

from typing import List
from typing_extensions import TypedDict

from langgraph.graph import END, StateGraph, START

from pprint import pprint

dir_source_path = "./data/tmp_txt"
loader = DirectoryLoader(
    f"./{dir_source_path}",
    glob="**/*.txt",
    loader_cls=TextLoader,  # Explicitly use TextLoader for .txt files
)

docs = []
for doc in loader.load():
    source = doc.metadata["source"].strip(dir_source_path)
    title_rest, source = source.split("[")[:-1], source.split("[")[-1]
    video_title = "[".join(title_rest)
    video_id = source.split("]")[0]
    doc.metadata["video_id"] = video_id
    doc.metadata["video_title"] = video_title
    docs.append(doc)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
doc_splits = text_splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
retriever = FAISS.from_documents(doc_splits, embeddings).as_retriever(
    search_kwargs={"k": 20}
)
reranker = FlashrankRerank()

doc_return_cnt = 3
bm25_retriever = BM25Retriever.from_documents(doc_splits, k=doc_return_cnt)
faiss_vectorstore = FAISS.from_documents(doc_splits, embeddings)
faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": doc_return_cnt})

hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever], weights=[0.7, 0.3]
)

llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0.8,
    num_predict=1024,
)


class SubQuery(BaseModel):
    """Given a user question, break it down into distinct sub questions that \
    you need to answer in order to answer the original question."""

    questions: List[str] = Field(description="The list of sub questions")


sub_question_generator = llm.with_structured_output(SubQuery)

prompt = hub.pull("rlm/rag-prompt")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = prompt | llm | StrOutputParser()


# Data model
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


retrieval_grader = llm.with_structured_output(GradeDocuments)
system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | retrieval_grader


class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


hallucination_grader = llm.with_structured_output(GradeHallucinations)
system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

hallucination_grader = hallucination_prompt | hallucination_grader


class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


generation_grader = llm.with_structured_output(GradeAnswer)
system = """You are a grader assessing whether an answer addresses / resolves a question \n 
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

answer_grader = answer_prompt | generation_grader
system = """You a question re-writer that converts an input question to a better version that is optimized \n 
     for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate an improved question.",
        ),
    ]
)

question_rewriter = re_write_prompt | llm | StrOutputParser()


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    sub_questions: List[str]
    generation: str
    documents: List[str]


def decompose(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---QUERY DECOMPOSITION ---")
    question = state["question"]

    # Reranking
    sub_queries = sub_question_generator.invoke(question)
    return {"sub_questions": sub_queries.questions, "question": question}


def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    sub_questions = state["sub_questions"]
    question = state["question"]

    # Retrieval
    documents = []
    for sub_question in sub_questions:
        docs = hybrid_retriever.get_relevant_documents(sub_question)
        documents.extend(docs)
    return {"documents": documents, "question": question}


def rerank(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RERANK---")
    question = state["question"]
    documents = state["documents"]

    # Reranking
    documents = reranker.compress_documents(query=question, documents=documents)
    return {"documents": documents, "question": question}


def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    for d in documents:
        attepmts = 0
        score = None
        while isinstance(score, type(None)) and attepmts < 3:
            try:
                score = retrieval_grader.invoke(
                    {"question": question, "document": d.page_content},
                    {"recursion_limit": 100},
                )
            except Exception as e:
                print(f"Error: {e}")
                attepmts += 1
                continue
        grade = score.binary_score
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
    return {"documents": filtered_docs, "question": question}


def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]

    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}


def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    state["question"]
    filtered_documents = state["documents"]

    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
        )
        return "transform_query"
    # We have relevant documents, so generate answer
    print("---DECISION: GENERATE---")
    return "generate"


def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    attepmts = 0
    score = None
    while isinstance(score, type(None)) and attepmts < 3:
        try:
            score = hallucination_grader.invoke(
                {"documents": documents, "generation": generation}
            )
        except Exception as e:
            print(f"Error: {e}")
            attepmts += 1
            continue

    print("score", score)
    grade = score.binary_score

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score.binary_score
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
        return "not useful"
    pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
    return "not supported"


workflow = StateGraph(GraphState)

workflow.add_node("decompose", decompose)  # query decompostion
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("rerank", rerank)  # rerank
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generatae
workflow.add_node("transform_query", transform_query)  # transform_query

workflow.add_edge(START, "decompose")
workflow.add_edge("decompose", "retrieve")
workflow.add_edge("retrieve", "rerank")
workflow.add_edge("rerank", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "transform_query",
    },
)

app = workflow.compile()
question = "Who is Machine Learning Engineer?"
inputs = {"question": question}
for output in app.stream(inputs, {"recursion_limit": 50}):
    for key, value in output.items():
        pprint(f"Node '{key}':")
        print(value)
    pprint("\n---\n")


pprint(value["generation"])
documents = value["documents"]
unique_sources = list(
    set(
        [
            (
                doc.metadata.get("video_title", "No title found"),
                doc.metadata.get("video_id", "No source found"),
            )
            for doc in documents
        ]
    )
)
for title, v_id in unique_sources:
    print(f"Title: {title} \nURL: https://www.youtube.com/watch?v={v_id}")
