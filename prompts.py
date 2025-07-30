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
