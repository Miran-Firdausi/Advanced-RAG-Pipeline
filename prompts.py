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
