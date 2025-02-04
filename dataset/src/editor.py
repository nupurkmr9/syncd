from typing import Optional

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import FluxPosEmbed, apply_rotary_emb
from einops import rearrange

axes_dims_rope = (16, 56, 56)
pos_embed = FluxPosEmbed(theta=10000, axes_dim=axes_dims_rope)

def otsu(mask_in):
    # normalize
    mask_norm = (mask_in - mask_in.min(-1, keepdim=True)[0]) / \
       (mask_in.max(-1, keepdim=True)[0] - mask_in.min(-1, keepdim=True)[0])
    
    N = 10
    bs = mask_in.shape[0]
    h = mask_in.shape[1]
    mask = []
    for i in range(bs):
        threshold_t = 0.
        max_g = 0.
        for t in range(N):
            mask_i = mask_norm[i]
            low = mask_i[mask_i < t/N]
            high = mask_i[mask_i >= t/N]
            low_num = low.shape[0]/h
            high_num = high.shape[0]/h
            low_mean = low.mean()
            high_mean = high.mean()
        
            g = low_num*high_num*((low_mean-high_mean)**2)
            if g > max_g:
                max_g = g
                threshold_t = t/N
            
        mask_i[mask_i < threshold_t] = 0
        mask_i[mask_i > threshold_t] = 1
        mask.append(mask_i)
    mask_out = torch.stack(mask, dim=0)
            
    return mask_out


class AttentionStore:
    def __init__(self, token_indices, num_att_layers, WIDTH):
        super().__init__()
        self.num_att_layers = num_att_layers
        self.token_indices = token_indices
        self.cur_att_layer = 0
        self.cur_step = 0
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.attention_maps = None
        self.attention_mask = None
        self.WIDTH = WIDTH

    @staticmethod
    def get_empty_store():
        return {"mixed": [], "single": [],}
    
    def __call__(self, q, k, v, is_cross, place_in_unet, num_heads, scale):
        self.forward(q, k, v, is_cross, place_in_unet, num_heads, scale)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.after_step()
        return

    def show_cross_attention(self, res=32, from_where=['mixed', 'single']):
        avg_attn = self.get_average_attention()
        attention_maps = []
        for each in from_where:
            attention_maps += avg_attn[each]
        attention_maps = torch.stack([x for x in attention_maps if x.shape[1] == res*res], 0)
        batch = attention_maps.size(1)
        self.attention_maps = []
        for i in range(batch):
            attention_map = attention_maps.mean(0)[i].unsqueeze(0)
            attention_map = otsu((attention_map[:, :, self.token_indices[i][0]: self.token_indices[i][1]]).sum(-1)) # > 0.65*attention_map[:, :, self.ref_token_idx].max()
            self.attention_maps.append(attention_map)
        self.attention_maps = torch.stack(self.attention_maps)

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0
        self.attention_store = {}
        self.step_store = self.get_empty_store()

    def after_step(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.show_cross_attention(res=self.WIDTH // 16, from_where=['mixed', 'single'])
        self.step_store = self.get_empty_store()
        ## After each step update the self.attention_mask used in the shared attention processor
        hw = (self.WIDTH // 16) * (self.WIDTH // 16)
        kernel_tensor = torch.ones((1, 1, 3, 3), dtype=self.attention_maps.dtype, device=self.attention_maps.device)
        self.attention_mask = torch.clamp(torch.nn.functional.conv2d(self.attention_maps.to(self.attention_maps.dtype).reshape(-1, 1, self.WIDTH // 16, self.WIDTH // 16), kernel_tensor, padding='same'), 0, 1)
        NUM = self.attention_mask.shape[0]
        self.attention_mask = torch.cat([rearrange(self.attention_mask, "(b n) c h w -> b 1 n c h w", n=NUM)] +
                                    [torch.roll(rearrange(self.attention_mask, "(b n) c h w -> b 1 n c h w", n=NUM), shifts=i, dims=2) for i in range(1,NUM)], dim=1)
        self.attention_mask = rearrange(
            self.attention_mask, "b n1 n c h w-> (b n) (n1 h w) c"
        )
        self.attention_mask = torch.cat([torch.ones_like(self.attention_mask[:, :512, :]), self.attention_mask], 1)
        self.attention_mask = torch.einsum("b i d, b j d -> b i j", torch.ones_like(self.attention_mask[:, :512 + hw]), self.attention_mask)
        self.attention_mask[:, :512+hw, :512+hw] = 1
        self.attention_mask[:, :512, 512+hw:] = 0
        self.attention_mask = self.attention_mask.masked_fill(self.attention_mask == 0, -65504.0)
        self.attention_mask = rearrange(
                            self.attention_mask.unsqueeze(0).expand(24, -1, -1, -1),
                            "nh b ... -> b nh ..."
                        )

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention

    def forward(self, q, k, v, is_cross, place_in_transformer, num_heads, scale):
        if is_cross:
            sim = (
                torch.einsum("b i d, b j d -> b i j", q[:, :512], k) * scale
            ) 
            sim = sim[:, :512, 512:].permute(0, 2, 1).softmax(dim=-1)
            key = f"{place_in_transformer}"
            self.step_store[key].append(rearrange(sim, "(b h) n d -> b h n d", h=num_heads).mean(1))

        return


class SharedAttnProc(torch.nn.Module):
    def __init__(
        self,
        attn_op,
        selfattn=True,
        single_transformer=False,
        NUM=3,
    ) -> None:
        super().__init__()
        self.attn_op = attn_op
        self.selfattn = selfattn
        self.single_transformer = single_transformer
        self.NUM = NUM

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        txt_ids: Optional[torch.Tensor] = None,
        img_ids_concat: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        editor=None,
    ):
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        txt_seq = 512 # number of tokens for the text prompt
        end_of_hidden_states = hidden_states.shape[1]
        if self.selfattn:
            if not self.single_transformer:
                hidden_states = torch.cat([hidden_states]+ 
                                          [rearrange(torch.roll(rearrange(hidden_states, "(b n) hw c -> b n hw c", n=self.NUM), shifts=i, dims=1), "b n hw c -> (b n) hw c") for i in range(1,self.NUM)], dim=1)
            
            else:
                hidden_states = torch.cat([hidden_states]+ 
                                          [rearrange(torch.roll(rearrange(hidden_states[:, txt_seq:], "(b n) hw c -> b n hw c", n=self.NUM), shifts=i, dims=1), "b n hw c -> (b n) hw c") for i in range(1,self.NUM)], dim=1)
            
                
            query = attn.to_q(hidden_states[:, :end_of_hidden_states])
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
    
            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads
    
            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if attn.norm_q is not None:
                query = attn.norm_q(query)
            if attn.norm_k is not None:
                key = attn.norm_k(key)
    
            # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
            if encoder_hidden_states is not None:
                # `context` projections.
                encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
                encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
                encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)
    
                encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                    batch_size, -1, attn.heads, head_dim
                ).transpose(1, 2)
                encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                    batch_size, -1, attn.heads, head_dim
                ).transpose(1, 2)
                encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                    batch_size, -1, attn.heads, head_dim
                ).transpose(1, 2)
    
                if attn.norm_added_q is not None:
                    encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
                if attn.norm_added_k is not None:
                    encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)
                query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
                key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
                value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

            if image_rotary_emb is not None:
                ids = torch.cat((txt_ids, img_ids_concat), dim=0)
                image_rotary_emb_global = pos_embed(ids)
        
                query = apply_rotary_emb(query, image_rotary_emb)
                key = apply_rotary_emb(key, image_rotary_emb_global)


            if editor is not None:
                _ = editor(rearrange(query, "b nh ... -> (b nh) ..."), 
                           rearrange(key, "b nh ... -> (b nh) ...")[:, :end_of_hidden_states + (encoder_hidden_states.shape[1] if encoder_hidden_states is not None else 0)], 
                           rearrange(value, "b nh ... -> (b nh) ...")[:, :end_of_hidden_states + (encoder_hidden_states.shape[1] if encoder_hidden_states is not None else 0)], True, "single" if self.single_transformer else "mixed", attn.heads, scale=attn.scale)


            hidden_states = F.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=0.0,
                is_causal=False,
                attn_mask=(editor.attention_mask.to(query.dtype) if editor is not None else attention_mask.to(query.dtype)) if timestep[0] < 1 else None,
            )

            hidden_states = hidden_states.transpose(1, 2).reshape(
                batch_size, -1, attn.heads * head_dim
            )
            
            if encoder_hidden_states is not None:
                encoder_hidden_states, hidden_states = (
                    hidden_states[:, : encoder_hidden_states.shape[1]],
                    hidden_states[:, encoder_hidden_states.shape[1] : encoder_hidden_states.shape[1] + end_of_hidden_states],
                )
    
                # linear proj
                hidden_states = attn.to_out[0](hidden_states)
                # dropout
                hidden_states = attn.to_out[1](hidden_states)
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
                return hidden_states, encoder_hidden_states
            else:
                return hidden_states[:, :end_of_hidden_states]
        else:
            return self.attn_op(attn,
                        hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=attention_mask,
                        image_rotary_emb=image_rotary_emb,
                        )

