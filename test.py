import os

import openai
openai.api_key = os.getenv('OPENAI_API_KEY', '')

def get_response(gpt_model="gpt-4-1106-preview", temperature=0.1):
    response = openai.ChatCompletion.create(
        model=gpt_model,
        messages=message_log,
        temperature=temperature,
        max_tokens=4096,
        # functions=functions,
        # function_call='auto',
    )
    response_message = response["choices"][0]["message"]
    return response_message

def main():
    get_response()

if __name__ == "__main__":
    main()
