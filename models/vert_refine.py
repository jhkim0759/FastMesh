import trimesh

import torch
from torch import nn
from torch.nn import Module

from .models import register
from .pc_encoder import PointEmbed
from models.modules.transformers import AdaLNSelfAttn
from models.miche.encode import load_model

@register("VertRefine")
class VertRefine(Module):
    def __init__(self):
        super().__init__()

        self.cond_dim = 768
        self.feat_dim = 1024
        self.layer_num = 6
        self.cond_length = 257
        self.pad_id = -1

        self.point_encoder = load_model(ckpt_path=None)
        self.cond_head_proj = nn.Linear(self.cond_dim, self.feat_dim)
        self.cond_proj = nn.Linear(self.cond_dim * 2, self.feat_dim)
        
        dpr = [x.item() for x in torch.linspace(0, 0.1, self.layer_num)]
        self.offset_transformers = nn.ModuleList([AdaLNSelfAttn(
                    block_idx=idx, embed_dim=self.feat_dim, norm_layer=nn.LayerNorm, num_heads=8, mlp_ratio=4.,
                    drop=0, attn_drop=0, drop_path=dpr[idx], last_drop_p=0 if idx == 0 else dpr[idx-1],
                    attn_l2_norm=False, flash_if_available=True, fused_if_available=True, only_flash=True, use_scale=True
                ) for idx in range(self.layer_num)])
        self.offset_head = nn.Linear(self.feat_dim, 3)
        self.point_embed = PointEmbed(
            hidden_dim=48,
            dim = self.feat_dim,
        )
        self.train()
        
    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self,"point_encoder"):
            self.point_encoder.eval()
            for param in self.point_encoder.parameters():
                param.requires_grad = False

    def process_point_feature(self, point_feature):
        encode_feature = torch.zeros(point_feature.shape[0], self.cond_length, self.feat_dim,
                                    device=self.cond_head_proj.weight.device, dtype=self.cond_head_proj.weight.dtype)
        encode_feature[:, 0] = self.cond_head_proj(point_feature[:, 0])
        shape_latents = self.point_encoder.to_shape_latents(point_feature[:, 1:])
        encode_feature[:, 1:] = self.cond_proj(torch.cat([point_feature[:, 1:], shape_latents], dim=-1))

        return encode_feature

    def forward(
        self,
        data_dict,
        is_eval = False,
        point_feature = None,
        **kwargs
    ):
        if point_feature is None:
            point_feature = self.point_encoder.encode_latents(data_dict["pc_normal"])
        processed_point_feature = self.process_point_feature(point_feature=point_feature)

        BATCH = point_feature.shape[0]

        vertices = data_dict['vertices'].reshape(BATCH, -1, 3).to(point_feature.dtype)
        attention_mask = vertices[:,:,0] != self.pad_id
        sequence_max_length = attention_mask.sum(dim=1).max()

        vertices = vertices[:, :sequence_max_length]
        attention_mask = attention_mask[:, :sequence_max_length]

        input_embed = self.point_embed(vertices)
        hidden_state = torch.cat([processed_point_feature, input_embed],1) 

        pad_attention_mask = torch.ones((attention_mask.shape[0], self.cond_length), device=attention_mask.device, dtype=attention_mask.dtype)
        attention_mask = torch.concatenate((pad_attention_mask, attention_mask), dim=1)
        
        for layer in self.offset_transformers:
            hidden_state = layer(hidden_state, attention_mask.float())

        attention_mask = attention_mask[:,self.cond_length:]
        hidden_state = hidden_state[:,self.cond_length:]

        pred = self.offset_head(hidden_state)

        if is_eval:
            pred = pred.clamp(0,1)/128 + vertices
            mesh_list = []
            faces = data_dict['faces']
            for bid in range(BATCH):
                cur_vert = pred[bid]
                cur_vert = cur_vert[attention_mask[bid].bool()].reshape(-1,3)

                cur_faces = faces[bid]
                cur_faces = cur_faces[cur_faces[:,0]!=-1].reshape(-1,3)

                mesh = trimesh.Trimesh(vertices=cur_vert.cpu().numpy(), faces=cur_faces.cpu().numpy())
                mesh.merge_vertices()
                mesh.update_faces(mesh.nondegenerate_faces())
                mesh.update_faces(mesh.unique_faces())
                mesh.remove_unreferenced_vertices()
                mesh.fix_normals()

                mesh_list.append(mesh)

            return mesh_list

        else:
            loss = abs(pred[attention_mask.bool()] - gt[attention_mask.bool()]).mean()
            data_dict['loss'] = loss
            return data_dict
    
    def refine_vertices(self, vertices: list, point_feature: torch.Tensor):
        processed_point_feature = self.process_point_feature(point_feature=point_feature)
        BATCH = point_feature.shape[0]

        vertices = torch.nn.utils.rnn.pad_sequence(vertices, batch_first=True, padding_value=-1, padding_side='right')
        vertices = vertices.reshape(BATCH, -1, 3).to(point_feature.dtype).to(point_feature.device)
        attention_mask = vertices[:,:,0] != self.pad_id

        input_embed = self.point_embed(vertices)
        hidden_state = torch.cat([processed_point_feature, input_embed],1) 

        pad_attention_mask = torch.ones((attention_mask.shape[0], self.cond_length), device=attention_mask.device, dtype=attention_mask.dtype)
        attention_mask = torch.concatenate((pad_attention_mask, attention_mask), dim=1)
        
        for layer in self.offset_transformers:
            hidden_state = layer(hidden_state, attention_mask.float())

        attention_mask = attention_mask[:,self.cond_length:]
        hidden_state = hidden_state[:,self.cond_length:]

        pred = self.offset_head(hidden_state).clamp(0,1)/128

        vertices[attention_mask.bool()] += pred[attention_mask.bool()]
        
        return vertices

        
