from typing import Union
from fastapi import FastAPI

from configparser import ConfigParser




def load_model(config:ConfigParser):
    '''
    Load llm and tokenizer model based on the {config}
    '''
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    tokenizer = AutoTokenizer.from_pretrained(                  config["Paths"]["tokenizer_path"])
    
    # You need to install bitsandbytes, accelerator for load_in_4bit=true.
    # bitsandbytes dosen't support Windows default. You can use WSL or install the compiled version for Windows(links below).
    # https://github.com/jllllll/bitsandbytes-windows-webui
    model = AutoModelForCausalLM.from_pretrained(               config["Paths"]["model_path"],
                                                load_in_4bit =  config["Model Options"]["load_in_4bit"],
                                                device_map   =  config["Model Options"]["device_map"])
    _=model.eval()
    
    return (tokenizer, model)


def generate(tokenizer, model, text:str, do_sample=False, top_k=3, top_p=0.95, temperature=0.8, max_new_tokens=64, num_return_sequences=1):
    # Tokenize input text, and send it to cuda=0
    tokens = tokenizer(text, return_tensors="pt").to(0)
    
    input_ids = tokens.input_ids
    attention_mask = tokens.attention_mask
    
    answer = model.generate(
                    input_ids=input_ids, 
                    attention_mask = attention_mask,
                    do_sample = do_sample,
                    top_k = top_k,
                    top_p=top_p,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=num_return_sequences,
                    eos_token_id=2,
                    pad_token_id=2,
                    early_stopping=True
                   )
    
    # Decode model output
    final = tokenizer.batch_decode(answer)[0]
    
    # Clean up text.
    return clean_up_for_chang(final)

def clean_up_for_chang(text:str) -> str:
    #while '<|endoftext|><|endoftext|>' in text: text = text.replace('<|endoftext|><|endoftext|>', '<|endoftext|>')
    text = text.replace('<|endoftext|>', '')
    return text


def make_prompt_for_chang(history:list) -> str:
    '''
    history: History of communication.
    ---
    return prompt for llm model.
    '''
    prompt = ''
    for line in history:
        prompt += f'{line[0]}:{line[1]}\n'
        
        # ADD eos token at the end of the bot's answer.
        if (line[0]=='B'): prompt+='<|endoftext|>'
    
    return prompt

def get_answer(question:str, history:list=[], *args, **kwargs):
    
    # Append the question to the chat history
    history.append(['A', question])
    
    # Get answer and append it to the chat history.
    answer = generate(tokenizer, model, make_prompt_for_chang(history)).split('\nA:')[-1]
    history.append(['B', answer])
    
    return answer, history


# if not __name__== '__main__':
#     print('this is not the main')
#     import os
#     os.system("pause")
#     exit()


# Get configs
config = ConfigParser()
config.read('config.ini')

# Load models
tokenizer, model = load_model(config)

# Start api server
app = FastAPI()


@app.get("/")
def hello_world():
    return {"Hello": "World"}

@app.get("/chat/{question}")
def chat(question:str, history:list=[]):
    
    # Get answer and updated chat history
    answer, history = get_answer(question, history)
    
    # Return it. "answer" should be str type.
    return {"answer": f"{answer}",
            "history": history}
