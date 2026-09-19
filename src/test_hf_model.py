
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading tokenizer...", flush=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...", flush=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=(
        torch.float16 if device == "cuda"
        else torch.float32
    ),
)

model.to(device)
model.eval()

messages = [
    {
        "role": "system",
        "content": "You are a helpful research assistant."
    },
    {
        "role": "user",
        "content": "Explain the Transformer architecture in simple terms."
    }
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(
    prompt,
    return_tensors="pt"
).to(device)

print("Generating answer...", flush=True)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

answer_tokens = outputs[0][inputs["input_ids"].shape[1]:]

answer = tokenizer.decode(
    answer_tokens,
    skip_special_tokens=True
)

print("\nModel answer:\n")
print(answer)