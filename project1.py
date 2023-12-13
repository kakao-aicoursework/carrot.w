import csv
import json
import os
from typing import List

import openai
import tkinter as tk
import pandas as pd
from tkinter import scrolledtext


from pandas import DataFrame

openai.api_key = os.getenv('OPENAI_API_KEY', '')
pd.set_option('display.max_columns', None)


def append_to_csv(row: List[str], file_name: str, header: List[str]):
    file_exists = os.path.isfile(file_name)

    with open(file_name, 'a') as f_object:
        writer = csv.writer(f_object)

        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)


def extract_keywords(keywords: str, original_text: str, file_to_save: str):
    _HEADERS=["keywords", "text"]
    append_to_csv([keywords, original_text], file_to_save, _HEADERS)
    return

def find_most_similar_rows(row_index: int, df: DataFrame):
    return df.iloc[[row_index]]["text"].item()

def send_message(message_log, functions, gpt_model="gpt-4-1106-preview", temperature=0.1, default_value ={}):
    response = openai.ChatCompletion.create(
        model=gpt_model,
        messages=message_log,
        temperature=temperature,
        functions=functions,
        function_call='auto',
        max_tokens=4096,
    )

    response_message = response["choices"][0]["message"]

    if response_message.get("function_call"):
        function_name = response_message["function_call"]["name"]
        function_to_call = globals().get(function_name)
        function_args = json.loads(response_message["function_call"]["arguments"])
        function_args.update(default_value)
        # 사용하는 함수에 따라 사용하는 인자의 개수와 내용이 달라질 수 있으므로
        # **function_args로 처리하기
        function_response = function_to_call(**function_args)
    return function_response

def read_text_file(file_path):
    with open(file_path, 'r') as file:
        # Read the entire content of the file
        content = file.read()
        return content


def preprocess(file_path):
    # preprocess data into tabular data.
    system_prompt = {
            "role": "system",
            "content": "Extract 3-5 main keywords from the text.\n"
    }

    functions = [
        {
            "name": "extract_keywords",
            "description": "Parse 3-5 keywords from given keywords.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keywords":
                        {
                            "type": "string",
                            "description": "Parse 3-5 multiple keywords from the text. Keywords are separated by comma.",
                        }

                },
                "required": ["keywords"]
            }
        }
    ]

    content = read_text_file(file_path)
    csv_file_name = file_path.replace(".txt", ".csv")
    paragrphs = content.split("##")
    for paragraph in paragrphs[1:]:
        print(paragraph.strip())
        user_prompt ={
            "role": "user",
            "content": f"{paragraph.strip()}\n."
        }

        response = openai.ChatCompletion.create(
            model="gpt-4-1106-preview",
            messages=[system_prompt, user_prompt],
            temperature=0.2,
            functions=functions,
            function_call="auto",
            max_tokens=4096,
        )

        function_call_response = response["choices"][0]["message"].get("function_call", {})
        if function_call_response.get("name") == "extract_keywords":
            keywords = json.loads(function_call_response["arguments"])["keywords"]
            extract_keywords(keywords, paragraph.strip(), csv_file_name)



def main():
    # Do preprocess only once.
    original_text_file = "./data/project_data_카카오톡채널.txt"
    # preprocess("./data/project_data_카카오톡채널.txt")

    preprocessed_file = "./data/project_data_카카오톡채널.csv"
    df = pd.read_csv(preprocessed_file)

    # make system prompt
    keywords_list = ""
    for idx, keywords in enumerate(df['keywords']):
        keywords_list += f"- row {idx}: {keywords}\n"

    message_log = [
        {
            "role": "system",
            "content": f"Find the number of row which has most similar keywords with user query from the keyword list.\n" + keywords_list

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
        response = send_message(message_log, functions, default_value=dict(df=df))
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
