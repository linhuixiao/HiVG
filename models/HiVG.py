import torch
import torch.nn as nn
import torch.nn.functional as F
from .vl_transformer import build_vl_transformer

from .clip import *
from torchvision.transforms import Resize
from utils.misc import (NestedTensor, nested_tensor_from_tensor_list)
from collections import OrderedDict

# import bitsandbytes as bnb
from transformers import CLIPModel, CLIPConfig, CLIPTextConfig, CLIPTextModel, CLIPVisionConfig, CLIPVisionModel
from transformers import CLIPTokenizer, AutoTokenizer, CLIPImageProcessor
from peft import get_peft_config, PeftModel, get_peft_model, LoraConfig, TaskType
from torch.nn.parameter import Parameter
from typing import Any, Optional, Tuple, Union
import math


class Modified_CLIPVisionEmbeddings(nn.Module):
    def __init__(self, clip_embed):
        super().__init__()
        self.config = clip_embed.config
        self.embed_dim = clip_embed.embed_dim
        self.image_size = clip_embed.image_size
        self.patch_size = clip_embed.patch_size
        self.class_embedding = clip_embed.class_embedding  # 768
        self.patch_embedding = clip_embed.patch_embedding
        self.num_patches = clip_embed.num_patches
        self.num_positions = clip_embed.num_positions  # 197
        self.position_embedding = clip_embed.position_embedding  # 197 * 768
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)))

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]  # B C H W
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, width, grid, grid], B 768 H/16 W/16
        h, w = patch_embeds.shape[2], patch_embeds.shape[3]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)  # B L H
        class_embeds = self.class_embedding.expand(batch_size, 1, -1)  # B * 1 * 768
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)

        cls_pos = self.position_embedding.weight[0:1, :]
        abs_pos = self.position_embedding.weight[1:, :]  # 196 * 768
        xy_num = abs_pos.shape[0]
        assert xy_num == self.num_patches  # 196
        size = int(math.sqrt(xy_num))  # 14
        assert size * size == xy_num

        if size != h or size != w:
            new_abs_pos = F.interpolate(  # 1 14 14 768 --> 1 768 14 14 --> 1 768 40 40
                abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2),
                size=(h, w),
                mode="bicubic",
                antialias=True,
                align_corners=False,
            )
            new_abs_pos = new_abs_pos.permute(0, 2, 3, 1).reshape(1, h * w, -1)
            position_embedding = torch.cat([cls_pos.unsqueeze(0), new_abs_pos], dim=1)  # 1 1601 768
            embeddings = embeddings + position_embedding.repeat(batch_size, 1, 1)
        else:  # 14 == 14
            embeddings = embeddings + self.position_embedding(self.position_ids)

        return embeddings


class VisionEmbeddings(nn.Module):
    def __init__(self, clip_embed):
        super().__init__()
        self.config = clip_embed.config
        self.embed_dim = clip_embed.embed_dim
        self.image_size = clip_embed.image_size
        self.patch_size = clip_embed.patch_size
        self.class_embedding = clip_embed.class_embedding  # 768
        self.patch_embedding = clip_embed.patch_embedding
        self.num_patches = clip_embed.num_patches
        self.num_positions = clip_embed.num_positions  # 此时是197
        self.position_embedding = clip_embed.position_embedding  # 197 * 768
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)))

    def forward(self, pixel_values: torch.FloatTensor, position_embedding) -> torch.Tensor:
        batch_size = pixel_values.shape[0]  # B C H W
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, width, grid, grid], B 768 H/16 W/16
        h, w = patch_embeds.shape[2], patch_embeds.shape[3]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)  # B L H
        class_embeds = self.class_embedding.expand(batch_size, 1, -1)  # B * 1 * 768
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)

        embeddings = embeddings + position_embedding(self.position_ids)

        return embeddings


class ClassInstantier(OrderedDict):
    def __getitem__(self, key):
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        return cls(**kwargs)

#
# ACT2CLS = {
#     "gelu": GELUActivation,
#     "gelu_10": (ClippedGELUActivation, {"min": -10, "max": 10}),
#     "gelu_fast": FastGELUActivation,
#     "gelu_new": NewGELUActivation,
#     "gelu_python": (GELUActivation, {"use_gelu_python": True}),
#     "gelu_pytorch_tanh": PytorchGELUTanh,
#     "gelu_accurate": AccurateGELUActivation,
#     "laplace": LaplaceActivation,
#     "linear": LinearActivation,
#     "mish": MishActivation,
#     "quick_gelu": QuickGELUActivation,
#     "relu": nn.ReLU,
#     "relu2": ReLUSquaredActivation,
#     "relu6": nn.ReLU6,
#     "sigmoid": nn.Sigmoid,
#     "silu": SiLUActivation,
#     "swish": SiLUActivation,
#     "tanh": nn.Tanh,
# }
# ACT2FN = ClassInstantier(ACT2CLS)


class CLIP_Cross_Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        text_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(text_states), -1, bsz)
        value_states = self._shape(self.v_proj(text_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        # apply the causal_attention_mask first
        if causal_attention_mask is not None:
            if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                    f" {causal_attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped


class CLIPAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        # apply the causal_attention_mask first
        if causal_attention_mask is not None:
            if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                    f" {causal_attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class CLIPMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # self.activation_fn = ACT2FN[config.hidden_act]
        # self.activation_fn = nn.ReLU
        self.activation_fn = QuickGELU()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)  # 768, 3072
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class TOKEN_MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.activation_fn = QuickGELU()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class CLIPEncoderLayer_with_Crossmodal_Bridge(nn.Module):
    # def __init__(self, config: CLIPConfig):
    #     super().__init__()
    #     self.embed_dim = config.hidden_size
    #     self.self_attn = CLIPAttention(config)
    #     self.cross_attn = CLIP_Cross_Attention(config)
    #
    #     self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
    #     self.mlp = CLIPMLP(config)
    #     self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def __init__(self, args, i, clip_encoder_layer, config: CLIPConfig, adapt_layer, extract_text_layer, text_config):
        super().__init__()
        self.embed_dim = clip_encoder_layer.embed_dim
        self.self_attn = clip_encoder_layer.self_attn

        """ Multi-layer Adaptive Cross-modal Bridge """
        self.enable_adaptive_weights = args.enable_adaptive_weights
        if i in adapt_layer:
            self.cross_norm = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)  # eps=1e-05
            self.cross_attn = CLIP_Cross_Attention(config)
            self.cross_mlp = CLIPMLP(config)

            text_embed_dim = text_config.hidden_size  # 512 for base model, 768 for Large model
            self.cross_gate = nn.Linear(text_embed_dim * len(extract_text_layer), self.embed_dim)  # clip vision 768
            self.cross_adaptive_weights = nn.ModuleList([nn.Embedding(77, text_embed_dim) for i in range(len(extract_text_layer))])

        self.layer_norm1 = clip_encoder_layer.layer_norm1
        self.mlp = clip_encoder_layer.mlp
        self.layer_norm2 = clip_encoder_layer.layer_norm2

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer,
        adapt_layer,
        text_states,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        """ Multi-layer Adaptive Cross-modal Bridge """
        if layer in adapt_layer:
            residual = hidden_states
            hidden_states = self.cross_norm(hidden_states)
            if self.enable_adaptive_weights:
                adpt_text_states = []
                for i, embed_layer in enumerate(self.cross_adaptive_weights):
                    adaptive_weights = torch.mul(embed_layer.weight.unsqueeze(0).repeat(text_states[i].shape[0], 1, 1),
                                                 text_states[i].to(hidden_states.dtype))
                    adpt_text_state = adaptive_weights + text_states[i].to(hidden_states.dtype)
                    adpt_text_states.append(adpt_text_state)
                adpt_text_states = torch.cat(adpt_text_states, dim=-1).to(hidden_states.dtype)

                adpt_text_states = self.cross_gate(adpt_text_states)
                adpt_text_states = adpt_text_states.permute(1, 0, 2)  # B L H --> L B H
            else:
                adpt_text_states = text_states[-1].to(hidden_states.dtype).permute(1, 0, 2)

            hidden_states, attn_weights = self.cross_attn(
                hidden_states=hidden_states,
                text_states=adpt_text_states,
                attention_mask=attention_mask,
                causal_attention_mask=causal_attention_mask,
                output_attentions=output_attentions,
            )
            hidden_states = self.cross_mlp(hidden_states)
            hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class CLIPEncoder_with_Crossmodal_Bridge(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`CLIPEncoderLayer`].

    Args:
        config: CLIPConfig
    """

    def __init__(self, args, clip_encoder, adapt_layer, extract_text_layer, text_config):
        super().__init__()
        self.config = clip_encoder.config
        self.layers = nn.ModuleList([CLIPEncoderLayer_with_Crossmodal_Bridge(args, i, clip_encoder.layers[i], self.config,
                                                                             adapt_layer, extract_text_layer,
                                                                             text_config)
                                     for i in range(self.config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds,
        adapt_layer,
        text_states,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            causal_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Causal mask for the text model. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(encoder_layer),
                    hidden_states,
                    idx,
                    adapt_layer,
                    text_states,
                    attention_mask,
                    causal_attention_mask,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    idx,
                    adapt_layer,
                    text_states,
                    attention_mask,
                    causal_attention_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)

        return {"last_hidden_state": hidden_states, "hidden_states": encoder_states, "attentions": all_attentions}


class CLIP_Vision_Model_with_Crossmodal_Bridge(nn.Module):
    def __init__(self, args, clip_visu_model, adapt_layer, extract_text_layer, text_config):
        super().__init__()
        self.config = clip_visu_model.config
        self.embeddings = Modified_CLIPVisionEmbeddings(clip_visu_model.embeddings)
        self.pre_layrnorm = clip_visu_model.pre_layrnorm  # 原版代码拼错了
        self.encoder = CLIPEncoder_with_Crossmodal_Bridge(args, clip_visu_model.encoder, adapt_layer, extract_text_layer,
                                                          text_config)
        self.post_layernorm = clip_visu_model.post_layernorm

    # @add_start_docstrings_to_model_forward(CLIP_VISION_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPVisionConfig)
    def forward(
        self,
        adapt_layer,
        text_states,
        reg_src,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layrnorm(hidden_states)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            adapt_layer=adapt_layer,
            text_states=text_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs["last_hidden_state"]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return {
            "last_hidden_state": last_hidden_state,
            "pooler_output": pooled_output,
            "hidden_states": encoder_outputs["hidden_states"],
            "attentions": encoder_outputs["attentions"],
        }


"""
   HiVG is implemented on the basis of CLIP-VG, Github: https://github.com/linhuixiao/CLIP-VG
"""


class HiVG(nn.Module):
    def __init__(self, args):
        super(HiVG, self).__init__()
        print("init HiVG model...")
        if (args.model == "ViT-L/14-336"):
            print("init CLIP ViT-L/14-336")
            self.clip = CLIPModel.from_pretrained("/path_to_clip/clip-vit-large-patch14-336")
            self.extract_vision_layer = [12, 16, 20, 24]  # v4
            self.adapt_layer = [11, 15, 19, 23]
            self.patch_size = 14
        elif (args.model == "ViT-L/14"):  # main large model
            print("init CLIP ViT-L/14")
            self.clip = CLIPModel.from_pretrained("/path_to_clip/clip-vit-large-patch14")
            self.extract_vision_layer = [6, 12, 18, 24]  # final 版本
            self.adapt_layer = [] if args.warmup is True else [4, 10, 16, 22]  # large model is trained on two phrases
            self.patch_size = 14
        elif (args.model == "ViT-B/32"):
            print("init CLIP ViT-B/32")
            self.clip = CLIPModel.from_pretrained("/path_to_clip/clip-vit-base-patch32")
            self.extract_vision_layer = [1, 4, 8, 12]
            self.adapt_layer = [0, 3, 7, 11]
            self.patch_size = 32
        else:  # default base model
            print("init CLIP ViT-B/16")
            self.clip = CLIPModel.from_pretrained("/path_to_clip/clip-vit-base-patch16")
            """
             Note that there is no mistake here. Note that [1, 4, 8, 12], [0, 3, 7, 11] are the same layer.
             In the internal implementation of transformers, the index at vision branch [0] is the original
             image embedding. 
            """
            self.extract_vision_layer = [1, 4, 8, 12]
            self.adapt_layer = [0, 3, 7, 11]
            self.patch_size = 16

        # set extract_text_layer
        self.mixup_pretrain = args.mixup_pretrain
        if self.mixup_pretrain:
            self.extract_text_layer = [12]
        else:
            if args.dataset == "gref_umd" or args.dataset == "gref":
                self.extract_text_layer = [i+1 for i in range(12)]
            elif args.dataset == "unc+":
                self.extract_text_layer = [6, 12]
            elif args.dataset == "unc":
                self.extract_text_layer = [12]
            elif args.dataset == "referit":
                self.extract_text_layer = [6, 12]
            else:
                self.extract_text_layer = [12]

        print("\nextract vision layer: ", self.extract_vision_layer)
        print("extract text layer: ", self.extract_text_layer)
        print("image size: ", args.imsize, " * ", args.imsize)

        for parameter in self.clip.parameters():
            parameter.requires_grad_(False)

        print("adapt_layer: ", self.adapt_layer)
        self.clip.vision_model = CLIP_Vision_Model_with_Crossmodal_Bridge(args, self.clip.vision_model,
                                                                          self.adapt_layer, self.extract_text_layer,
                                                                          self.clip.text_model.config)

        """
            Note that the essence of the HiLoRA mechanism is a process of decomposing parameter learning, and its
            effectiveness is influenced by the learning rate and the number of epochs. Therefore, HiLoRA requires
            different learning rates and numbers of epochs at various stages for specific model configurations.
            If you do not need to enable HiLoRA, simply leave args.hi_lora_stage=0 as the default.
        """
        self.set_HiLoRA(args)

        if True:
            print("Open Multi-layer Adaptive Cross-modal Bridge parameters ...")
            for name, param in self.clip.vision_model.encoder.layers.named_parameters():
                if "cross_attn" in str(name).split(".") or "cross_norm" in str(name).split(".") \
                        or "cross_mlp" in str(name).split(".") or "cross_gate" in str(name).split(".") \
                        or "cross_adaptive_weights" in str(name).split("."):
                    print("param name: ", name)
                    param.requires_grad_(True)
            self.clip.print_trainable_parameters()

        self.hidden_dim = self.clip.projection_dim  # base model 512，large model 768
        self.imsize = args.imsize
        clip_visu_hidden_dim = self.clip.vision_model.config.hidden_size  # 768
        self.visu_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.condition_text_proj = nn.Linear(self.hidden_dim, clip_visu_hidden_dim)  # clip vision 768
        self.ml_text_feat_perceiver = nn.Linear(self.clip.text_embed_dim * len(self.extract_text_layer), clip_visu_hidden_dim)
        self.text_proj = nn.Linear(self.hidden_dim, self.hidden_dim)  # clip vision 768
        self.reg_token = nn.Embedding(1, self.hidden_dim)

        # divisor = 16
        self.num_visu_token = int((args.imsize / self.patch_size) ** 2)
        self.num_text_token = args.max_query_len
        num_total = self.num_visu_token + 1 + self.num_text_token + 1  # v token + [cls]token + t token + [REG]token
        self.vl_pos_embed = nn.Embedding(num_total, self.hidden_dim)

        self.vl_transformer = build_vl_transformer(args)

        self.bbox_embed = MLP(self.hidden_dim, self.hidden_dim, 4, 3)
        self.reg_pos_embed = nn.Embedding(1, self.hidden_dim)
        self.condition_text_pos_embed = nn.Embedding(self.num_text_token, clip_visu_hidden_dim)
        self.ml_visual_projection = nn.Linear(len(self.extract_vision_layer) * self.clip.vision_model.config.hidden_size,
                                              self.hidden_dim)
        self.ml_visual_projection.weight = nn.Parameter(torch.cat([self.clip.visual_projection.weight for i
                                                                   in range(len(self.extract_vision_layer))], dim=1))

        self.visu_token_norm = nn.LayerNorm(self.hidden_dim, eps=1e-05)  # 512, eps=1e-05
        self.visu_token_mlp = TOKEN_MLP(self.hidden_dim, 3072)  # 3072

        # TODO：Segmentation head for Referring Image Segmentation task. RIS works only when the seg mask is used.
        #  seg conv, 10GB, 14*14 --> 28*28 --> 56*56 --> 112*112
        hidden_dim = self.hidden_dim
        self.seg_conv1 = nn.ConvTranspose2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(2, 2), stride=(2, 2),
                                            padding=(0, 0), output_padding=(0, 0), bias=False)  # bias=False
        self.seg_conv2 = nn.ConvTranspose2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(2, 2), stride=(2, 2),
                                            padding=(0, 0), output_padding=(0, 0), bias=False)  # bias=False
        self.seg_conv3 = nn.ConvTranspose2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(2, 2), stride=(2, 2),
                                            padding=(0, 0), output_padding=(0, 0), bias=False)  # bias=False

    def set_HiLoRA(self, args):
        open_lora = True
        close_lora_parameter_update = False
        close_lora_vision_parameter_update = False
        close_lora_text_parameter_update = False

        if open_lora:
            target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc_in", "fc_out", "wte"]
            peft_config = LoraConfig(target_modules=target_modules, inference_mode=False, r=32, lora_alpha=16,
                                     lora_dropout=0.1, bias='none')

            # lora stage 0 for MSCOCO debais
            self.clip = get_peft_model(self.clip, peft_config)
            self.clip.print_trainable_parameters()
            for parameter in self.clip.parameters():
                parameter.requires_grad_(False)
            self.clip.print_trainable_parameters()

            if args.hi_lora_stage == 1:
                print("open lora stage 1")
                self.clip = get_peft_model(self.clip, peft_config)
                self.clip.print_trainable_parameters()
                for parameter in self.clip.vision_model.parameters():
                    parameter.requires_grad_(False)

                for name, param in self.clip.vision_model.encoder.layers.named_parameters():
                    if "lora_A" in str(name).split(".") or "lora_B" in str(name).split("."):
                        if "0" in str(name).split(".") or "1" in str(name).split(".") or "2" in str(name).split(".") \
                                or "3" in str(name).split(".") or "4" in str(name).split("."):
                            print("param name: ", name)
                            param.requires_grad_(True)
                    else:
                        param.requires_grad_(False)
                self.clip.print_trainable_parameters()

            if args.hi_lora_stage == 2:
                # stage 1:
                print("open lora stage 1")
                self.clip = get_peft_model(self.clip, peft_config)
                self.clip.print_trainable_parameters()
                for parameter in self.clip.parameters():
                    parameter.requires_grad_(False)
                self.clip.print_trainable_parameters()

                # stage 2:
                print("open lora stage 2")
                self.clip = get_peft_model(self.clip, peft_config)
                self.clip.print_trainable_parameters()
                for parameter in self.clip.vision_model.parameters():
                    parameter.requires_grad_(False)

                for name, param in self.clip.vision_model.encoder.layers.named_parameters():
                    if "lora_A" in str(name).split(".") or "lora_B" in str(name).split("."):
                        if "0" in str(name).split(".") or "1" in str(name).split(".") or "2" in str(name).split(".") \
                                or "3" in str(name).split(".") or "4" in str(name).split(".") \
                                or "5" in str(name).split(".") or "6" in str(name).split(".") \
                                or "7" in str(name).split("."):
                            print("param name: ", name)
                            param.requires_grad_(True)
                    else:
                        param.requires_grad_(False)
                self.clip.print_trainable_parameters()

            if args.hi_lora_stage == 3:
                # stage 1:
                print("open lora stage 1")
                self.clip = get_peft_model(self.clip, peft_config)
                self.clip.print_trainable_parameters()
                for parameter in self.clip.parameters():
                    parameter.requires_grad_(False)
                self.clip.print_trainable_parameters()

                # stage 2:
                print("open lora stage 2")
                self.clip = get_peft_model(self.clip, peft_config)
                self.clip.print_trainable_parameters()
                for parameter in self.clip.parameters():
                    parameter.requires_grad_(False)
                self.clip.print_trainable_parameters()

                # stage 3:
                print("open lora stage 3")
                self.clip = get_peft_model(self.clip, peft_config)
                self.clip.print_trainable_parameters()

                for parameter in self.clip.vision_model.parameters():
                    parameter.requires_grad_(False)

                for name, param in self.clip.vision_model.encoder.layers.named_parameters():
                    if "lora_A" in str(name).split(".") or "lora_B" in str(name).split("."):
                        if "0" in str(name).split(".") or "1" in str(name).split(".") or "2" in str(name).split(".") \
                                or "3" in str(name).split(".") or "4" in str(name).split(".") \
                                or "5" in str(name).split(".") or "6" in str(name).split(".") \
                                or "7" in str(name).split(".") \
                                or "8" in str(name).split(".") or "9" in str(name).split(".") \
                                or "10" in str(name).split(".") or "11" in str(name).split("."):
                            print("param name: ", name)
                            param.requires_grad_(True)
                    else:
                        param.requires_grad_(False)
                self.clip.print_trainable_parameters()

            if close_lora_parameter_update:
                for parameter in self.clip.parameters():
                    parameter.requires_grad_(False)
                self.clip.print_trainable_parameters()
            else:
                if close_lora_vision_parameter_update:
                    for parameter in self.clip.vision_model.parameters():
                        parameter.requires_grad_(False)
                    self.clip.print_trainable_parameters()
                if close_lora_text_parameter_update:
                    for parameter in self.clip.text_model.parameters():
                        parameter.requires_grad_(False)
                    self.clip.print_trainable_parameters()

    def tensorize_inputs(self, images: NestedTensor, texts: NestedTensor):
        image_tensors = images.tensors
        texts_tensors = texts.tensors

        return image_tensors, texts_tensors

    def get_masks(self, images: NestedTensor, texts: NestedTensor):
        # torch_resize = Resize([14, 14])
        torch_resize = Resize([int(self.imsize / self.patch_size), int(self.imsize / self.patch_size)])  # 14 * 14 = 196， or， 16 * 16 = 256
        visu_masks = torch_resize(images.mask)
        visu_masks = visu_masks.to(torch.bool)
        visu_masks = visu_masks.flatten(1)  # visu_mask：B*L, torch.Size([B, 196])
        # text mask follow bert process
        # text_masks = texts.mask.to(torch.bool)
        # text_masks = ~text_masks
        # text_masks = text_masks.flatten(1)
        # assert text_masks is not None

        return visu_masks

    def encode_text(self, text_data, device=None):
        text_tensors = clip.tokenize(text_data, context_length=77, truncate=True).to(device)  # 4 * 77
        text_mask = text_tensors.eq(0).bool()  # 4 * 77, The ones that need masking are 1.
        return text_tensors, text_mask

    def forward(self, img_data, text_data):
        batch_size = img_data.tensors.shape[0]  # 得到batch_size
        image_tensors = img_data.tensors
        text_tensors, text_mask = self.encode_text(text_data, img_data.tensors.device)

        clip_text_features = self.clip.text_model(text_tensors, output_attentions=True, output_hidden_states=True,
                                                  return_dict=True)  # B * 77 * 512
        text_features = self.clip.text_projection(clip_text_features.last_hidden_state)
        text_eos_embed = self.clip.text_projection(clip_text_features.pooler_output)  # torch.Size([64, 512])

        if self.mixup_pretrain:
            ml_text_features = [self.condition_text_proj(text_features.float())]
        else:
            ml_text_features = [clip_text_features.hidden_states[i] for i in self.extract_text_layer]

        visu_mask = self.get_masks(img_data, text_data)

        # target regression token
        reg_src = self.reg_token.weight.unsqueeze(0).repeat(batch_size, 1, 1)  # B * 1 * hidden_dim

        clip_image_features = self.clip.vision_model(self.adapt_layer, ml_text_features, reg_src, image_tensors,
                                                     output_attentions=True, output_hidden_states=True,
                                                     return_dict=True)  # B * 197 * 512
        attention_map = clip_image_features["attentions"]  # tuple, used for draw the attention map

        ml_image_features = [clip_image_features["hidden_states"][i] for i in self.extract_vision_layer]
        img_cls_embed = self.clip.visual_projection(clip_image_features["pooler_output"])  # torch.Size([64, 512])

        ml_image_features = torch.cat(ml_image_features, dim=2)
        image_features = self.ml_visual_projection(ml_image_features)

        visu_src = self.visu_proj(image_features.float())  # (N*B)xC
        text_src = self.text_proj(text_features.float())  # B * 77 * 512

        # permute BxLenxC to LenxBxC
        visu_src = visu_src.permute(1, 0, 2)  # 197 * 4 * 512
        text_src = text_src.permute(1, 0, 2)  # 77 * 4 * 512
        reg_src = reg_src.permute(1, 0, 2)  # 1 * B * 512

        vl_src = torch.cat([reg_src, visu_src, text_src], dim=0)

        # mask
        reg_mask = torch.zeros((batch_size, 1)).to(reg_src.device).to(torch.bool)
        cls_mask = torch.zeros((batch_size, 1)).to(reg_src.device).to(torch.bool)
        vl_mask = torch.cat([reg_mask, cls_mask, visu_mask, text_mask], dim=1)
        vl_pos = self.vl_pos_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)

        vg_hs = self.vl_transformer(vl_src, vl_mask, vl_pos)  # (1+L+N)xBxC
        box_hs = vg_hs[0]
        pred_box = self.bbox_embed(box_hs).sigmoid()

        # normalized features
        img_cls_embed = img_cls_embed / img_cls_embed.norm(p=2, dim=-1, keepdim=True)
        text_eos_embed = text_eos_embed / text_eos_embed.norm(p=2, dim=-1, keepdim=True)
        # cosine similarity as logits
        logit_scale = self.clip.logit_scale.exp()
        logits_per_text = torch.matmul(text_eos_embed, img_cls_embed.t()) * logit_scale
        logits_per_image = logits_per_text.t()

        # visual token align
        vg_hs_visu_features = vg_hs[2: 2 + self.num_visu_token].permute(1, 0, 2)  # B L H
        clip_last_layer_features = self.visu_token_mlp(self.visu_token_norm(vg_hs_visu_features))

        vg_hs_text = vg_hs[2 + self.num_visu_token:].permute(1, 0, 2)
        vg_hs_text_eos_embed = vg_hs_text[torch.arange(vg_hs_text.shape[0]), text_tensors.argmax(dim=-1)]
        vg_hs_text_eos_embed = vg_hs_text_eos_embed / vg_hs_text_eos_embed.norm(p=2, dim=-1, keepdim=True)

        visu_token_similarity = torch.mul(vg_hs_text_eos_embed.unsqueeze(1).repeat(1, self.num_visu_token, 1),
                                          clip_last_layer_features)  # torch.Size([96, 196, 512])
        visu_token_similarity = visu_token_similarity.sum(axis=-1, keepdim=False)  # torch.Size([96, 196])

        patch_num = int(math.sqrt(vg_hs_visu_features.shape[1]))
        channel = vg_hs_visu_features.shape[2]
        assert patch_num * patch_num == vg_hs_visu_features.shape[1]
        seg_features = vg_hs_visu_features.permute(0, 2, 1).reshape(batch_size, channel, patch_num, patch_num)
        seg_features = self.seg_conv3(self.seg_conv2(self.seg_conv1(seg_features)))
        seg_features = seg_features.permute(0, 2, 3, 1)
        seg_mask = torch.mul(vg_hs_text_eos_embed.reshape(batch_size, 1, 1, vg_hs_text_eos_embed.shape[-1]).repeat(1, seg_features.shape[1], seg_features.shape[2], 1),
                             seg_features)
        seg_mask = seg_mask.sum(axis=-1, keepdim=False).unsqueeze(1)  # B 1 H W

        return pred_box, logits_per_text, logits_per_image, visu_token_similarity, seg_mask


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
