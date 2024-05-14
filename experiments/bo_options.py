def lora_target_map(model: str):
    match model:
        case 'microsoft/phi-2':
            return {
                'qkv_proj': ['k_proj', 'q_proj', 'v_proj'],
                'attn_head': ['k_proj', 'q_proj', 'v_proj', 'dense'],
                'attn_head_and_mlp': ['k_proj', 'q_proj', 'v_proj', 'dense', 'fc1', 'fc2'],
                'attn_head_and_mlp_with_Q': ['k_proj', 'q_proj', 'v_proj', 'dense', 'fc1', 'fc2', 'attn_shortcut_Q'],
                'attn_head_mlp_lm_head_with_Q': [
                    'k_proj',
                    'q_proj',
                    'v_proj',
                    'dense',
                    'fc1',
                    'fc2',
                    'attn_shortcut_Q',
                    'lm_head',
                ],
            }
        case 'facebook/opt-125m' | 'facebook/opt-1.3b' | 'facebook/opt-2.7b' | 'facebook/opt-6.7b' | 'facebook/opt-13b' | 'facebook/opt-30b' | 'facebook/opt-66b':
            return {
                'qkv_proj': ['k_proj', 'q_proj', 'v_proj'],
                'attn_head': ['k_proj', 'q_proj', 'v_proj', 'out_proj'],
                'attn_head_and_mlp': ['k_proj', 'q_proj', 'v_proj', 'out_proj', 'fc1', 'fc2'],
                'attn_head_and_mlp_with_Q': [
                    'k_proj',
                    'q_proj',
                    'v_proj',
                    'out_proj',
                    'attn_shortcut_Q',
                    'fc1',
                    'fc2',
                    'mlp_shortcut_Q',
                ],
                'attn_head_mlp_lm_head_with_Q': [
                    'k_proj',
                    'q_proj',
                    'v_proj',
                    'out_proj',
                    'attn_shortcut_Q',
                    'fc1',
                    'fc2',
                    'mlp_shortcut_Q',
                    'lm_head',
                ],
            }
        case 'meta-llama/Llama-2-7b-hf' | 'meta-llama/Llama-2-13b-hf' | 'meta-llama/Llama-2-70b-hf' | 'meta-llama/Meta-Llama-3-8B' | 'meta-llama/Meta-Llama-3-8B-Instruct' | 'meta-llama/Meta-Llama-3-70B' | 'meta-llama/Meta-Llama-3-70B-Instruct':
            return {
                'qkv_proj': ['k_proj', 'q_proj', 'v_proj'],
                'attn_head': ['k_proj', 'q_proj', 'v_proj', 'o_proj'],
                'attn_head_and_mlp': [
                    'k_proj',
                    'q_proj',
                    'v_proj',
                    'o_proj',
                    'gate_proj',
                    'up_proj',
                    'down_proj',
                ],
                'attn_head_and_mlp_with_Q': [
                    'k_proj',
                    'q_proj',
                    'v_proj',
                    'o_proj',
                    'attn_shortcut_Q',
                    'gate_proj',
                    'up_proj',
                    'down_proj',
                    'mlp_shortcut_Q',
                ],
                'attn_head_mlp_lm_head_with_Q': [
                    'k_proj',
                    'q_proj',
                    'v_proj',
                    'o_proj',
                    'attn_shortcut_Q',
                    'gate_proj',
                    'up_proj',
                    'down_proj',
                    'mlp_shortcut_Q',
                    'lm_head',
                ],
            }
        case 'microsoft/Phi-3-mini-4k-instruct':
            return {
                'qkv_proj': ['qkv_proj'],
                'attn_head': ['qkv_proj', 'o_proj'],
                'attn_head_and_mlp': [
                    'qkv_proj',
                    'o_proj',
                    'gate_up_proj',
                    'down_proj',
                ],
                'attn_head_and_mlp_with_Q': [
                    'qkv_proj',
                    'o_proj',
                    'attn_shortcut_Q',
                    'gate_up_proj',
                    'down_proj',
                    'mlp_shortcut_Q',
                ],
                'attn_head_mlp_lm_head_with_Q': [
                    'qkv_proj',
                    'o_proj',
                    'attn_shortcut_Q',
                    'gate_up_proj',
                    'down_proj',
                    'mlp_shortcut_Q',
                    'lm_head',
                ],
            }
        case _:
            raise RuntimeError(f'Lora target map undefined for model={model}')
