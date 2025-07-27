import aiohttp
from typing import List, Union, Optional
from tenacity import retry, wait_random_exponential, stop_after_attempt, wait_fixed
from typing import Dict, Any
from dotenv import load_dotenv
import os
from openai import AsyncOpenAI
import async_timeout
from transformers import AutoTokenizer

from AgentDropout.llm.format import Message
from AgentDropout.llm.price import cost_count, cost_count_llama3, cost_count_deepseek
from AgentDropout.llm.llm import LLM
from AgentDropout.llm.llm_registry import LLMRegistry

import torch


load_dotenv()
MINE_BASE_URL = os.getenv("MINE_BASE_URL")
MINE_API_KEYS = os.getenv("VLLM_API_KEY")

# print(MINE_BASE_URL)


# @retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(3))
# async def achat(
#     model: str,
#     msg: List[Dict],):
#     request_url = MINE_BASE_URL
#     authorization_key = MINE_API_KEYS
#     headers = {
#         'Content-Type': 'application/json',
#         'authorization': authorization_key
#     }
#     data = {
#         "name": model,
#         "inputs": {
#             "stream": False,
#             "msg": repr(msg),
#         }
#     }
#     async with aiohttp.ClientSession() as session:
#         async with session.post(request_url, headers=headers ,json=data) as response:
#             response_data = await response.json()
#             if isinstance(response_data['data'],str):
#                 prompt = "".join([item['content'] for item in msg])
#                 cost_count(prompt,response_data['data'],model)
#                 return response_data['data']
#             else:
#                 raise Exception("api error")

@retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(3))
async def achat(model: str, msg: List[Dict],):
    api_kwargs = dict(api_key = MINE_API_KEYS, base_url = MINE_BASE_URL)
    aclient = AsyncOpenAI(**api_kwargs)
    try:
        async with async_timeout.timeout(1000):
            completion = await aclient.chat.completions.create(model=model,messages=msg)
        response_message = completion.choices[0].message.content
        
        if isinstance(response_message, str):
            prompt = "".join([item['content'] for item in msg])
            cost_count(prompt, response_message, model)
            return response_message

    except Exception as e:
        raise RuntimeError(f"Failed to complete the async chat request: {e}")

# @retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(6))
async def achat_deepseek(model: str, msg: List[Dict],):
    model = ''
    # print(1111111)
    api_kwargs = dict(api_key = deepseek_api, base_url = deepseek_url)
    aclient = AsyncOpenAI(**api_kwargs)
    try:
        async with async_timeout.timeout(1000):
            completion = await aclient.chat.completions.create(model=model,messages=msg)
        # print(completion)
        response_message = completion.choices[0].message.content
        
        if isinstance(response_message, str):
            prompt = "".join([item['content'] for item in msg])
            cost_count_deepseek(prompt, response_message, model)
            return response_message

    except Exception as e:
        raise RuntimeError(f"Failed to complete the async chat request: {e}")

@retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(3))
@retry(wait=wait_fixed(2), stop=stop_after_attempt(5))
async def achat_llama(model: str, msg: List[Dict]):
    # print(111111111111)
    api_kwargs = dict(api_key = MINE_API_KEYS, base_url = MINE_BASE_URL)
    aclient = AsyncOpenAI(**api_kwargs)
    try:
        async with async_timeout.timeout(1000):
            completion = await aclient.chat.completions.create(model=model,messages=msg)
        response_message = completion.choices[0].message.content
        
        if isinstance(response_message, str):
            prompt = "".join([item['content'] for item in msg])
            cost_count_llama3(prompt, response_message, model)
            return response_message

    except Exception as e:
        print(f"Error in achat_llama: {e}")
        # raise

# from huggingface_hub import AsyncInferenceClient
# from tenacity import retry, wait_fixed, stop_after_attempt
# import asyncio
# from typing import List, Dict


# from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
# from transformers import modeling_utils

# @retry(wait=wait_fixed(2), stop=stop_after_attempt(2))
# async def achat_llama(model, tokenizer, messages):
#     try:
#         text = tokenizer.apply_chat_template(
#                             messages,
#                             tokenize=False,
#                             add_generation_prompt=True
#                         )

#         inputs = tokenizer([text], return_tensors="pt").to(model.device)

#         prompt_tokens = inputs['input_ids'].shape[1]

#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=1024,
#             temperature=0.7,
#             return_dict_in_generate=True
#         )

#         generated_tokens = outputs.sequences[0, inputs['input_ids'].shape[1]:]
#         completion_tokens = generated_tokens.shape[0]
#         total_tokens = prompt_tokens + completion_tokens
        
#         # Decode the generated text
#         generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
#         # Format the complete response
#         response = {
#             "choices": [{
#                 "message": {
#                     "content": generated_text
#                 }
#             }],
#             "usage": {
#                 "prompt_tokens": prompt_tokens,
#                 "completion_tokens": completion_tokens,
#                 "total_tokens": total_tokens
#             }
#         }
#         response_message = response["choices"][0]["message"]["content"]
#         return response_message

#     except Exception as e:
#         print(f"Error in achat_llama: {e}")
#         raise



# @retry(wait=wait_fixed(2), stop=stop_after_attempt(2))
# async def achat_llama(model: str, msg: List[Dict]):
#     # Initialize HuggingFace async client
#     print(f"model: {model}")
#     client = AsyncInferenceClient(
#         provider="novita",
#         model=model,
#         api_key="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",  # Replace with your HuggingFace token
#         timeout=1000
#     )
    
#     try:
#         # Convert messages to prompt format (adjust based on model requirements)
#         prompt = "\n".join([f"{m['role']}: {m['content']}" for m in msg])
#         print(f"prompt: {prompt}")
        
#         # Generate response
#         response = await client.chat.completions.create(
#             messages=msg,
#             max_tokens=1024,
#             temperature=0.7,
#         )

#         print(f"response: {response}")
        
#         # Cost tracking (you'll need to adapt this for HuggingFace)
#         prompt_text = "".join([item['content'] for item in msg])
#         cost_count_llama3(prompt_text, response, model)
        
#         return response

#     except Exception as e:
#         print(f"Error in achat_llama: {e}")
#         raise
    

@LLMRegistry.register('GPTChat')
class GPTChat(LLM):

    def __init__(self, model_name: str):
        self.model_name = model_name

    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS
        
        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]
        return await achat(self.model_name,messages)
    
    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        pass

@LLMRegistry.register('deepseek')
class DeepseekChat(LLM):

    def __init__(self, model_name: str):
        self.model_name = model_name

    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS
        
        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]
        return await achat_deepseek(self.model_name,messages)
    
    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        pass

@LLMRegistry.register('llama')
class LlamaChat(LLM):

    def __init__(self, model_name: str):
        self.model_name = model_name
        # print(11111111111111111111)
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     self.model_name, 
        #     device_map="cuda", 
        #     torch_dtype=torch.bfloat16,
        #     # trust_remote_code=True
        # )
        # print(f"Successfully loaded Hugging Face model: {self.model_name}")
        
        # self.model.eval() 
        # self.tokenizer = AutoTokenizer.from_pretrained(
        #         self.model_name, 
        #         # device_map="auto", 
        #         # torch_dtype="auto", 
        #         # trust_remote_code=True
        #     )

    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS
        
        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]
        # return await achat_llama(self.model, self.tokenizer, messages)
        return await achat_llama(self.model_name, messages)

    
    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        pass