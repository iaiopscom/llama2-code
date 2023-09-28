from typing import Iterator
from llama_cpp import Llama
from huggingface_hub import hf_hub_download


def download_model():
    # See https://github.com/OpenAccess-AI-Collective/ggml-webui/blob/main/tabbed.py
    # https://huggingface.co/spaces/kat33/llama.cpp/blob/main/app.py
    print(f"Downloading model: {model_repo}/{model_filename}")
    file = hf_hub_download(
            repo_id=model_repo, filename=model_filename
    )
    print("Downloaded " + file)
    return file

#model_repo = "TheBloke/CodeLlama-7B-Instruct-GGUF"
model_repo = "TheBloke/CodeLlama-13B-Instruct-GGUF"
#model_filename = "codellama-7b-instruct.Q4_K_S.gguf"
model_filename = "codellama-13b-instruct.Q5_K_S.gguf"

model_path = download_model()

# load Llama-2
llm = Llama(model_path=model_path, n_ctx=4000, verbose=False)


def get_prompt(message: str, chat_history: list[tuple[str, str]],
               system_prompt: str) -> str:
    texts = [f'[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n']
    for user_input, response in chat_history:
        texts.append(f'{user_input.strip()} [/INST] {response.strip()} </s><s> [INST] ')
    texts.append(f'{message.strip()} [/INST]')
    return ''.join(texts)

def generate(prompt, max_new_tokens, temperature, top_p, top_k):
    return llm(prompt,
            max_tokens=max_new_tokens,
            stop=["</s>"],
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stream=False)


def get_input_token_length(message: str, chat_history: list[tuple[str, str]], system_prompt: str) -> int:
    prompt = get_prompt(message, chat_history, system_prompt)
    input_ids = llm.tokenize(prompt.encode('utf-8'))
    return len(input_ids)


def run(message: str,
        chat_history: list[tuple[str, str]],
        system_prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50) -> Iterator[str]:
    prompt = get_prompt(message, chat_history, system_prompt)
    output = generate(prompt, max_new_tokens, temperature, top_p, top_k)
    yield output['choices'][0]['text']

    # outputs = []
    # for resp in streamer:
    #     outputs.append(resp['choices'][0]['text'])
    #     yield ''.join(outputs)
