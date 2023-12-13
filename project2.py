import csv
import json
import os
from typing import List

import openai
import tkinter as tk
import pandas as pd
from tkinter import scrolledtext

from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.embeddings import SentenceTransformerEmbeddings, OpenAIEmbeddings, CacheBackedEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from pandas import DataFrame

openai.api_key = os.getenv('OPENAI_API_KEY', '')
pd.set_option('display.max_columns', None)

MAX_CNT_DOCS = 2
PROMPT_TEMPLATE = """You are a professional developer. Using below content, make an answer to the question.
1. Extract context which are related to the question.
2. Explain it kindly and in detail. If there are tabular data, make into plain words.\n
3. If there are some steps to follow, describe it step by step.\n
Here is context: \n
{context}
Here is question:
{question}
"""

def get_similar_docs(db, query: str):
    retriever = db.as_retriever(search_type="mmr")

    # query
    docs = retriever.get_relevant_documents(query)
    similar_docs = []

    for doc in docs:
        content = doc.page_content
        if len(content) < 20: # pass short paragraphs.
            continue
        similar_docs.append(content)
        if len(similar_docs)==MAX_CNT_DOCS:
            break
    return "\n".join(similar_docs)

def prompt_router(input):
    return PromptTemplate.from_template(input)



def preprocess(data_path: str):
    raw_documents = TextLoader(data_path).load()[0].page_content
    headers_to_split = [
        ("#", "header 1"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split)
    markdown_splits = markdown_splitter.split_text(raw_documents)

    # make embedding
    embeddings_model = OpenAIEmbeddings()
    store = LocalFileStore("./cache/")
    cached_embedder = CacheBackedEmbeddings.from_bytes_store(embeddings_model, store, namespace="project2")
    db = FAISS.from_documents(markdown_splits, cached_embedder)
    return db

def send_message(query, db, ):
    chain = (
        {"query": RunnablePassthrough()}
        | RunnablePassthrough.assign(question=query, context=get_similar_docs(db, query))
        # | RunnableLambda(prompt_router)
        # | ChatOpenAI()
        # | StrOutputParser()
    )
    output = chain.invoke(query)
    print(output)
    return output



def main():
    # Do preprocess only once.
    data_path = "./data/project_data_카카오싱크.txt"
    db = preprocess(data_path)


    message_log = [
        {
            "role": "system",
            "content": f"Find the number of row which has most similar keywords with user query from the keyword list.\n"

        }
    ]

    functions = [
        {
            "name": "find_most_similar_rows",
            "description": "Get the most similar rows using row index.",
            "parameters": {
                "type": "object",
                "properties": {
                    "row_index": {
                        "type": "integer",
                        "description": "Index of row which have the most similar meaning.",
                    },
                },
                "required": ["row_index"],
            },
        }
    ]

    def show_popup_message(window, message):
        popup = tk.Toplevel(window)
        popup.title("")

        # 팝업 창의 내용
        label = tk.Label(popup, text=message, font=("맑은 고딕", 12))
        label.pack(expand=True, fill=tk.BOTH)

        # 팝업 창의 크기 조절하기
        window.update_idletasks()
        popup_width = label.winfo_reqwidth() + 20
        popup_height = label.winfo_reqheight() + 20
        popup.geometry(f"{popup_width}x{popup_height}")

        # 팝업 창의 중앙에 위치하기
        window_x = window.winfo_x()
        window_y = window.winfo_y()
        window_width = window.winfo_width()
        window_height = window.winfo_height()

        popup_x = window_x + window_width // 2 - popup_width // 2
        popup_y = window_y + window_height // 2 - popup_height // 2
        popup.geometry(f"+{popup_x}+{popup_y}")

        popup.transient(window)
        popup.attributes('-topmost', True)

        popup.update()
        return popup

    def on_send():
        user_input = user_entry.get()
        user_entry.delete(0, tk.END)

        if user_input.lower() == "quit":
            window.destroy()
            return
        message_log.append({"role": "user", "content": f"- user query: {user_input}"})
        conversation.config(state=tk.NORMAL)  # 이동
        conversation.insert(tk.END, f"You: {user_input}\n", "user")  # 이동
        thinking_popup = show_popup_message(window, "처리중...")
        window.update_idletasks()
        # '생각 중...' 팝업 창이 반드시 화면에 나타나도록 강제로 설정하기
        response = send_message(user_input, db, )
        thinking_popup.destroy()

        # message_log.append({"role": "assistant", "content": response})

        # 태그를 추가한 부분(1)
        conversation.insert(tk.END, f"gpt assistant: {response}\n", "assistant")
        conversation.config(state=tk.DISABLED)
        # conversation을 수정하지 못하게 설정하기
        conversation.see(tk.END)

    window = tk.Tk()
    window.title("GPT AI")

    font = ("맑은 고딕", 10)

    conversation = scrolledtext.ScrolledText(window, wrap=tk.WORD, bg='#f0f0f0', font=font, fg='black')
    # width, height를 없애고 배경색 지정하기(2)
    conversation.tag_configure("user", background="#c9daf8")
    # 태그별로 다르게 배경색 지정하기(3)
    conversation.tag_configure("assistant", background="#e4e4e4")
    # 태그별로 다르게 배경색 지정하기(3)
    conversation.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    # 창의 폭에 맞추어 크기 조정하기(4)

    input_frame = tk.Frame(window)  # user_entry와 send_button을 담는 frame(5)
    input_frame.pack(fill=tk.X, padx=10, pady=10)  # 창의 크기에 맞추어 조절하기(5)

    user_entry = tk.Entry(input_frame)
    user_entry.pack(fill=tk.X, side=tk.LEFT, expand=True)

    send_button = tk.Button(input_frame, text="Send", command=on_send)
    send_button.pack(side=tk.RIGHT)

    window.bind('<Return>', lambda event: on_send())
    window.mainloop()


if __name__ == "__main__":
    main()
