from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

CHAT_HISTORY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Use the below context to answer the question.",
        ),
        ("system", "{context}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

different_perspective_template = """You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Original question: {question}"""

DIFFERENT_PERSPECTIVE_PROMPT = ChatPromptTemplate.from_template(
    different_perspective_template
)

CLARIFY_USER_QUERY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a smart assistant. You will help interpret a user's vague question by understanding the context of the document.",
        ),
        (
            "human",
            "Document Summary:\n{summary}\n\nUser's Question:\n{question}\n\nRewrite this question so that it is clear, unambiguous, and captures what the user is most likely asking based on the document.",
        ),
    ]
)

BASIC_PROMPT = f"""
"Answer the following question using short, clear, and descriptive statements. Focus on directly addressing the query based on the provided context. Avoid unnecessary elaboration, but ensure the core information is complete and accurate."

Examples Using the Template:
Q. What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?
A. A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.

Note: Reply with just the statement and nothing else.

<Context>:
{{context}}
<ContextEnd>

Question:
{{question}}

Answer:
"""

SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Summarize the following document in a concise and clear manner.",
        ),
        (
            "human",
            "Document:\n{doc_text}\n\nPlease provide a summary of the above document.",
        ),
    ]
)


SIMPLIFY_USER_QUERY = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a smart assistant. Your job is to take user's query and simplify it by removing specificity while preserving the core intent of the question, and rewriting to keep only the most relevant part of the question. Remove specific detail and make the question general.",
        ),
        (
            "human",
            "Document Summary:\n{summary}\n\nUser's Question:\n{question}\n\nRewrite this question so that it captures what the user is most likely asking based on the document.",
        ),
    ]
)
