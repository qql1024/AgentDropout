from huggingface_hub import InferenceClient

def achat_llama(model):
    # Initialize HuggingFace async client
    print(f"model: {model}")
    client = InferenceClient(
        model=model,
        token="",  # Replace with your HuggingFace token
        timeout=1000
    )
    
    try:
        # Convert messages to prompt format (adjust based on model requirements)
        prompt = "How are you?"
        print(f"prompt: {prompt}")
        
        # Generate response
        response = client.text_generation(
            prompt=prompt,
            max_new_tokens=1024,
            temperature=0.7,
            return_full_text=False
        )

        print(f"response: {response}")
        
        return response

    except Exception as e:
        print(f"Error in achat_llama: {e}")
        raise

model = "Qwen/Qwen2.5-7B-Instruct"
# achat_llama(model)

messages = [
    {
        "role": "user",
        "content": "How are you?",
    }
]
client = InferenceClient(
    provider="novita",
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    api_key="",
)

print(client.chat.completions.create(messages))