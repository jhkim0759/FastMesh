import trimesh 
import functools
import numpy as np
import torch
import torch.nn as nn
import networkx as nx
from models.miche.encode import load_model
from models.modules.transformers import AdaLNSelfAttn
from torch.utils.checkpoint import checkpoint
from models.models import register


@register("FaceGen")
class FaceGen(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.cond_length = 257
        self.cond_dim = 768
        self.feat_dim = 256*3 # 256*3 
        self.layerdrop = 0.1
        self.layer_num = args['layer_num']
        self.num_heads= 8

        self.threshold = args['face_threshold']
        self.n_discrete_size = args['n_discrete_size']
        self.pad_id = -1
        self.max_length = args['max_vertices']
   
        self.coord_embed = nn.Embedding(self.n_discrete_size+1, self.feat_dim//3, padding_idx=self.n_discrete_size)
        self.xyz_embed = nn.Embedding(4, self.feat_dim//3, padding_idx=3)
        self.vert_proj = nn.Linear(self.feat_dim, self.feat_dim)

        self.cond_proj = nn.Linear(self.cond_dim * 2, self.feat_dim)
        self.cond_head_proj = nn.Linear(self.cond_dim, self.feat_dim)
        
        dpr = [x.item() for x in torch.linspace(0, self.layerdrop, self.layer_num)]
        self.layers = nn.ModuleList([AdaLNSelfAttn(
                    block_idx=idx, embed_dim=self.feat_dim, norm_layer=nn.LayerNorm, num_heads=self.num_heads, mlp_ratio=4.,
                    drop=0, attn_drop=0, drop_path=dpr[idx], last_drop_p=0 if idx == 0 else dpr[idx-1],
                    attn_l2_norm=False, flash_if_available=True, fused_if_available=True, only_flash=True, use_scale=True
                ) for idx in range(self.layer_num)])
            
        self.grad_checkpointing = True
        gradient_checkpointing_kwargs = {"use_reentrant": False}
        self.gradient_checkpoint = functools.partial(checkpoint, **gradient_checkpointing_kwargs)
        
        
        self.last_dim = 32
        self.hidd_dim = 16
        self.last_head = nn.Linear(self.feat_dim, self.last_dim*6, bias=False)
        self.refine_layer = nn.Sequential(nn.Linear(self.last_dim, self.hidd_dim),
                                          nn.GELU(),
                                          nn.Linear(self.hidd_dim, 1, bias=False))
        
        self.post_process = False

        self.asloss = AsymmetricLoss(gamma_neg=3, gamma_pos=0, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True, scaler=1e4)

        self.init_weights()

        self.point_encoder = load_model(ckpt_path=None)
        if hasattr(self,"point_encoder"):
            self.point_encoder.eval()
            for param in self.point_encoder.parameters():
                param.requires_grad = False


    def init_weights(self, init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=0.02):
        import math 
        
        if init_std < 0: init_std = (1 / self.C / 3) ** 0.5     # init_std < 0: automated
        
        print(f'[init_weights] {type(self).__name__} with {init_std=:g}')
        for m in self.modules():
            with_weight = hasattr(m, 'weight') and m.weight is not None
            with_bias = hasattr(m, 'bias') and m.bias is not None
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if with_bias: m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if m.padding_idx is not None: m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if with_weight: m.weight.data.fill_(1.)
                if with_bias: m.bias.data.zero_()

        depth = len(self.layers)
        for block_idx, sab in enumerate(self.layers):
            sab.attn.proj.weight.data.div_(math.sqrt(2 * depth))
            sab.ffn.fc2.weight.data.div_(math.sqrt(2 * depth))
            if hasattr(sab.ffn, 'fcg') and sab.ffn.fcg is not None:
                nn.init.ones_(sab.ffn.fcg.bias)
                nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
            if hasattr(sab, 'ada_lin'):
                sab.ada_lin[-1].weight.data[2*self.C:].mul_(init_adaln)
                sab.ada_lin[-1].weight.data[:2*self.C].mul_(init_adaln_gamma)
                if hasattr(sab.ada_lin[-1], 'bias') and sab.ada_lin[-1].bias is not None:
                    sab.ada_lin[-1].bias.data.zero_()
            elif hasattr(sab, 'ada_gss'):
                sab.ada_gss.data[:, :, 2:].mul_(init_adaln)
                sab.ada_gss.data[:, :, :2].mul_(init_adaln_gamma)
    
    def train(self, mode: bool = True):
        super().train(mode)
        self.point_encoder.eval()

    def process_point_feature(self, point_feature):
        encode_feature = torch.zeros(point_feature.shape[0], self.cond_length, self.feat_dim,
                                    device=self.cond_head_proj.weight.device, dtype=self.cond_head_proj.weight.dtype)
        encode_feature[:, 0] = self.cond_head_proj(point_feature[:, 0])
        shape_latents = self.point_encoder.to_shape_latents(point_feature[:, 1:])
        encode_feature[:, 1:] = self.cond_proj(torch.cat([point_feature[:, 1:], shape_latents], dim=-1))

        return encode_feature

    def forward(self, 
                data_dict: dict, 
                is_eval: bool = False
                ) -> dict:
                
        point_feature = self.point_encoder.encode_latents(data_dict["pc_normal"])
        processed_point_feature = self.process_point_feature(point_feature=point_feature)
        BATCH, *_ = point_feature.shape

        with torch.no_grad():
            assert "sequence" in data_dict
            input_ids = data_dict['sequence'].reshape(BATCH, -1, 3) 
            xyz_ids = torch.tensor([0,1,2]).reshape(1, 1, 3).repeat(input_ids.shape[0],input_ids.shape[1],1).to(input_ids.device) # Batch, N Vert, 3

            attention_mask = input_ids[:,:,0] != self.pad_id 
            sequence_max_length = int(attention_mask.sum(dim=1).max())

            input_ids      = input_ids[:, :sequence_max_length]
            xyz_ids        = xyz_ids[:, :sequence_max_length]
            attention_mask = attention_mask[:, :sequence_max_length]

            input_ids[~attention_mask] = self.n_discrete_size

            xyz_ids[~attention_mask] = 3
            if not is_eval:
                gt_matrix = data_dict['matrix'][:, :sequence_max_length,:sequence_max_length]
                gt_matrix[gt_matrix<0.99] = 0.0

        # add cond_length to attention mask
        pad_attention_mask = torch.ones((BATCH, self.cond_length), device=attention_mask.device, dtype=attention_mask.dtype)
        attention_mask = torch.concatenate((pad_attention_mask, attention_mask), dim=1).float()
        train_mask = attention_mask[:,self.cond_length:].unsqueeze(-1) @ attention_mask[:,self.cond_length:].unsqueeze(-2)
        train_mask = train_mask.float().triu(diagonal=1)


        vert_embed = self.coord_embed(input_ids) + self.xyz_embed(xyz_ids)
        vert_embed = vert_embed.reshape(BATCH,-1, self.feat_dim)
        vert_embed = self.vert_proj(vert_embed) 
        hidden_state = torch.cat([processed_point_feature,vert_embed],1) 

        # Bidirectional Transformer
        for idx, layer in enumerate(self.layers):
            if self.grad_checkpointing and self.training:
                hidden_state = self.gradient_checkpoint(
                    layer.__call__,
                    hidden_state, 
                    attention_mask)
            else:
                hidden_state = layer(
                    hidden_state,
                    attention_mask)
        
        hidden_state   =   hidden_state[:,self.cond_length:]
        attention_mask = attention_mask[:,self.cond_length:]
        hidden_state[~attention_mask.bool()] = 0


        hidden_state = self.last_head(hidden_state)
        
        hidden_state = hidden_state.reshape(BATCH,sequence_max_length,self.last_dim,-1)
        hidden_state1, hidden_state2 = torch.chunk(hidden_state, 2, -1)
        hidden_state1 = hidden_state1 / (torch.norm(hidden_state1,dim=-1,keepdim=True) + 1e-6)
        hidden_state2 = hidden_state2 / (torch.norm(hidden_state2,dim=-1,keepdim=True) + 1e-6)

        hidden_state11, hidden_state12 = hidden_state1.unsqueeze(1), hidden_state1.unsqueeze(2)
        hidden_state21, hidden_state22 = hidden_state2.unsqueeze(1), hidden_state2.unsqueeze(2)

        matrix = (hidden_state11-hidden_state12).pow(2).sum(-1) - (hidden_state21-hidden_state22).pow(2).sum(-1)
        similarity_matrix_ = self.refine_layer(matrix[train_mask.bool()]).flatten()
        
        if is_eval:
            matrix = torch.zeros_like(train_mask).to(similarity_matrix_.dtype)
            matrix[train_mask.bool()] = similarity_matrix_.sigmoid()
            adj_matrix = matrix + matrix.transpose(-2,-1)      
            adj_matrix = adj_matrix>self.threshold
            
            mesh_list = []

            for batch_id in range(BATCH):
                vertices = data_dict['vertices'][batch_id][:sequence_max_length]
                pad_mask = vertices[:,0] != self.pad_id

                cur_matrix = (adj_matrix[batch_id][pad_mask][:, pad_mask]).triu(diagonal=1)
                sim_matrix = cur_matrix + cur_matrix.transpose(-2,-1)

                faces = self.adj_matrix_to_faces(sim_matrix)

                mesh = trimesh.Trimesh(vertices=vertices[pad_mask].cpu().numpy(), faces=faces)
                try:
                    mesh.merge_vertices()
                    mesh.update_faces(mesh.nondegenerate_faces())
                    mesh.update_faces(mesh.unique_faces())
                    mesh.remove_unreferenced_vertices()
                    mesh.fix_normals()
                except:
                    print(f"MESH PROCESS ERROR FACE IS {len(faces)}")
                
                mesh_list.append(mesh)
            return mesh_list
        
        else:
            as_loss = self.asloss(similarity_matrix_.sigmoid().float(), gt_matrix[train_mask.bool()].float())/BATCH
            data_dict['loss'] = as_loss
            return data_dict

    def adj_matrix_to_faces(self, adj_matrix):
        np_matrix = adj_matrix.int().cpu().numpy()
        if len(np_matrix.shape)>2:
            size=np_matrix.shape[-2]
            np_matrix = np_matrix.reshape(size,size)
        graph = nx.from_numpy_array(np_matrix)

        faces = postprocess(graph)

        return np.array(faces)
    

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-6, disable_torch_grad_focal_loss=True, scaler=1e4):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.scaler = scaler

    def forward(self, x_sigmoid, y):
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w
        return -loss.sum()/self.scaler

def postprocess(graph):
    faces = []
    for i, j in graph.edges:
        commons = sorted(list(nx.common_neighbors(graph, i, j)))
        for common in commons:
            if common<j: continue
            faces.append([i,j,common])

    return faces