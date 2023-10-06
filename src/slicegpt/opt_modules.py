import torch
from typing import List, Optional, Tuple, Union
from transformers.models.opt import OPTConfig
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.opt.modeling_opt import logger, OPTDecoder, OPTDecoderLayer


class CompressedOPTDecoderLayer(OPTDecoderLayer):
    """
    This class simulates the OPTDecoderLayer class from transformers
    but with the addition of a shortcut_Q attributes.
    We also support the input rotation and mean subtraction in this class (if needed).
    """

    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.register_buffer("mlp_shortcut_Q", None)
        self.register_buffer("attn_shortcut_Q", None)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`, *optional*): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )

        hidden_states = torch.nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        if self.attn_shortcut_Q is not None:
            rotated_shortcut = torch.matmul(residual, self.attn_shortcut_Q)
            hidden_states = rotated_shortcut + hidden_states
        else:
            hidden_states = residual + hidden_states

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            raise NotImplementedError(
                "Layer norm after attention is not implemented yet!"
            )
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Fully Connected
        hidden_states_shape = list(hidden_states.shape)
        hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)

        hidden_states = self.fc2(hidden_states)
        hidden_states = torch.nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )

        hidden_states_shape[
            -1
        ] = self.fc2.out_features  # to make sure the shape is correct

        if self.mlp_shortcut_Q is not None:
            rotated_shortcut = torch.matmul(residual, self.mlp_shortcut_Q)
            hidden_states = rotated_shortcut + hidden_states.view(hidden_states_shape)
        else:
            hidden_states = (residual + hidden_states).view(hidden_states_shape)

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
