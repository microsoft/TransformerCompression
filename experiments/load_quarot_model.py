# pip install transformers==4.41.2 flash_attn-2.5.8

import time
import json
import torch
from transformers import AutoTokenizer

from quarot.hf_utils import resolve_quarot_model_class


def load_quarot_model(model_name: str, checkpoint_path: str):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, use_fast=True, local_files_only=True)
    with open(f"{checkpoint_path}/args.json", "r") as f:
        args = json.load(f)

    lm_class = resolve_quarot_model_class(model_name)

    rotate = args['rotate']
    online_had_mlp = True if rotate else False
    online_had_attn = True if rotate else False
    rms_norm = True if rotate else False
    quarot_model = lm_class.from_pretrained(
        pretrained_model_name_or_path=checkpoint_path,
        online_had_mlp=online_had_mlp,
        online_had_attn=online_had_attn,
        rms_norm=rms_norm,
        act_bits=args.get('a_bits'),
        act_clip_ratio=args.get('a_clip_ratio'),
        k_bits=args.get('k_bits'),
        k_clip_ratio=args.get('k_clip_ratio'),
        k_groupsize=args.get('k_groupsize'),
        v_bits=args.get('v_bits'),
        v_clip_ratio=args.get('v_clip_ratio'),
        v_groupsize=args.get('v_groupsize'),
        local_files_only=True,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.float16
    ).to('cuda').eval()
    return quarot_model, tokenizer


save_dir = "C:/dev/phi3-mini/quarot_models_4bit"

model, tokenizer = load_quarot_model('microsoft/Phi-3-mini-4k-instruct', save_dir)

messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "What is the square root of 16?"},
]
template = "{% for message in messages %}{{'<|' + message['role'] + '|>' + '\n' + message['content'] + '<|end|>\n' }}{% endfor %}"
tokenizer.chat_template = template
tokenized_chat = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
).to('cuda')

start_time = time.time()
outputs = model.generate(tokenized_chat, max_new_tokens=200, eos_token_id=32007)  # 32007 corresponds to <|end|>
text = tokenizer.batch_decode(outputs)[0]
print("--- %s seconds ---" % (time.time() - start_time))
print(text)
