import os
import sys
import threading
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
)

from peft import PeftModel

# =========================
# Config
# =========================
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"     # change if you used a different base
ADAPTER_PATH = "C:\\Users\\dhnz2\\OneDrive\\Documents\\Github\\Rice-Acoustic-Sensor\\Codes\\finetune\\my_qwen"                 # folder that has adapter_config.json, adapter_model.bin
MERGE_ADAPTER = False                         # set True if you want a single merged model at load
MAX_NEW_TOKENS = 50
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.05

# =========================
# Helpers
# =========================
def get_device_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", torch.float16
    else:
        return "cpu", torch.float32

def ensure_tokens(tokenizer):
    # For many chat models, eos_token is set; but ensure pad exists
    if tokenizer.pad_token_id is None:
        # Safe default: use eos as pad
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.special_tokens_map.get("eos_token")
    return tokenizer

def format_with_chat_template(tokenizer, messages):
    """
    Use the model's chat template if available; otherwise fall back to a simple format.
    `messages` is a list of dicts: {"role": "system"/"user"/"assistant", "content": str}
    """
    apply = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,  # adds the assistant turn
        )
    # Fallback formatting
    lines = []
    for m in messages:
        role = m["role"].capitalize()
        lines.append(f"{role}: {m['content']}")
    lines.append("Assistant:")
    return "\n".join(lines)

def abs_path(p):
    return os.path.abspath(os.path.expanduser(p))

# =========================
# Load
# =========================
device, dtype = get_device_dtype()
adapter_dir = abs_path(ADAPTER_PATH)

print(f"• Device: {device} | DType: {dtype}\n• Adapter: {adapter_dir}")

# trust_remote_code=True is safe for Qwen family and avoids tokenizer/model quirks.
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer = ensure_tokens(tokenizer)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=dtype,
    device_map="auto" if device != "cpu" else None,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)

# Attach LoRA
model = PeftModel.from_pretrained(base_model, adapter_dir, torch_dtype=dtype)

# Optionally merge for single-weights inference
if MERGE_ADAPTER:
    print("Merging LoRA adapter into base weights... (one-time cost)")
    model = model.merge_and_unload()
    # after merge, model is a plain HF model; tokenizer is unchanged

model.eval()
torch.set_grad_enabled(False)

# =========================
# Chat loop (streaming)
# =========================
print("\n✅ Ready! Type 'end' to quit. Type 'clear' to reset the conversation.\n")

history = [
    {"role": "system", "content": "You are a helpful, concise assistant."}
]

while True:
    try:
        user_text = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n👋 Exiting.")
        break

    if not user_text:
        continue

    low = user_text.lower()
    if low in {"end", "quit", "exit", ":q"}:
        print("👋 Exiting.")
        break
    if low in {"clear", "/clear"}:
        history = history[:1]  # keep system
        print("↺ Conversation cleared.\n")
        continue

    # Add user turn
    history.append({"role": "user", "content": user_text})

    # Build model input using chat template
    prompt_text = format_with_chat_template(tokenizer, history)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device if device != "cpu" else "cpu")

    # Set up streamer to print tokens as they generate
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)

    gen_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        repetition_penalty=REPETITION_PENALTY,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Generate in background so we can stream output
    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    # Stream tokens
    sys.stdout.write("Assistant: ")
    sys.stdout.flush()
    generated_text = []
    for token_text in streamer:
        sys.stdout.write(token_text)
        sys.stdout.flush()
        generated_text.append(token_text)
    print("\n")  # newline after completion

    # Save assistant message to history
    history.append({"role": "assistant", "content": "".join(generated_text)})
