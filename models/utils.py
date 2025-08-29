import torch
import numpy as np
import scipy.stats as stats
from einops import repeat, reduce, rearrange, pack, unpack
import torch.nn as nn
import networkx as nx
import numbers

import torch
import torch.nn as nn


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0, input_dim=3):
    if i == -1:
        return nn.Identity(), input_dim

    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dim,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim

class RMSNorm(nn.Module):
    def __init__(self, dim, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()

        self.eps = eps

        if isinstance(dim, numbers.Integral):
            dim = (dim,)

        self.dim = torch.Size(dim)

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.weight = None

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        if self.weight is not None:
            # convert into half-precision if necessary
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                hidden_states = hidden_states.to(self.weight.dtype)
            hidden_states = hidden_states * self.weight
        else:
            hidden_states = hidden_states.to(input_dtype)

        return hidden_states.to(input_dtype)
    
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
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        # x_sigmoid = torch.sigmoid(x)
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

def random_masking(x, orders, mask_ratio_min=0.7):
    # generate token mask
    bsz, seq_len = orders.shape
    mask_rate = stats.truncnorm((mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25).rvs(1)[0]
    num_masked_tokens = int(np.ceil(seq_len * mask_rate))
    mask = torch.zeros(bsz, seq_len, device=x.device)
    mask = torch.scatter(mask, dim=-1, index=orders[:, :num_masked_tokens],
                            src=torch.ones(bsz, seq_len, device=x.device))
    return mask

def sample_orders(bsz, seq_len, shuffle=True):
    # generate a batch of random generation orders
    orders = []
    for _ in range(bsz):
        order = np.array(list(range(seq_len)))
        if shuffle: np.random.shuffle(order)

        orders.append(order)
    orders = torch.Tensor(np.array(orders)).long()
    return orders

def loop_detokenize(input_ids, n_max_triangles=1600, pad_id=-1, n_discrete_size=128):
        input_ids = input_ids.reshape(input_ids.shape[0], -1) # B x L
        batch_size = input_ids.shape[0]
        continuous_coors = torch.zeros((batch_size, n_max_triangles * 3 * 10, 3), device=input_ids.device)
        continuous_coors[...] = float('nan')
        for i in range(batch_size):
            cur_ids = input_ids[i]
            coor_loop_check = 0
            vertice_count = 0
            continuous_coors[i, :3, :] = torch.tensor([[-0.1, 0.0, 0.1], [-0.1, 0.1, 0.2], [-0.3, 0.3, 0.2]],
                                                      device=input_ids.device)
            error_judge = 0
            for id in cur_ids:
                if id == pad_id:
                    if coor_loop_check < 9:
                        error_judge=1
                    if coor_loop_check % 3 != 0:
                        error_judge=1
                    break
                elif id == n_discrete_size:
                    if coor_loop_check < 9:
                        error_judge=1
                        break
                    if coor_loop_check % 3 !=0:
                        error_judge=1
                        break
                    coor_loop_check = 0
                else:

                    if coor_loop_check % 3 == 0 and coor_loop_check >= 9:
                        continuous_coors[i, vertice_count] = continuous_coors[i, vertice_count-2]
                        continuous_coors[i, vertice_count+1] = continuous_coors[i, vertice_count-1]
                        vertice_count += 2
                    continuous_coors[i, vertice_count, coor_loop_check % 3] = undiscretize(id, -0.5, 0.5, n_discrete_size)
                    if coor_loop_check % 3 == 2:
                        vertice_count += 1
                    coor_loop_check += 1
            if vertice_count <= 3:
                error_judge=1

            if coor_loop_check % 3 != 0:
                error_judge=1

            if error_judge:
                continuous_coors[i, -1, -1] = 0

        continuous_coors = rearrange(continuous_coors, 'b (nf nv) c -> b nf nv c', nv=3, c=3)

        return continuous_coors # b, nf, 3, 3


def coor_discretize(
    t,
    continuous_range, # (-0.5, 0.5)
    num_discrete: int = 128
):
    lo, hi = continuous_range
    assert hi > lo

    t = (t - lo) / (hi - lo) # 0 <=t < 1
    t *= num_discrete # [0, num_discrete-1]
    assert (t - t.round()).sum() == 0
    assert (t <= num_discrete-1).all() and (t >= 0).all()  # 0 to num_discrete-1

    return t.long()

def undiscretize(
    t,
    low,#-0.5
    high,# 0.5
    num_discrete
):
    assert (t >= 0).all() and (t <= num_discrete-1).all()
    assert high>low
    t = t.float() #[0, num_discrete-1]

    t /= num_discrete  # 0<=t<1
    t = t * (high - low) + low # -0.5 <= t < 0.5
    assert (t < high).all() and (t >= low).all()
    return t

def mask_by_order(mask_len, order, bsz, seq_len):
    masking = torch.zeros(bsz, seq_len).cuda()
    masking = torch.scatter(masking, dim=-1, index=order[:, :mask_len.long()], src=torch.ones(bsz, seq_len).cuda()).bool()
    return masking
    

    
from itertools import combinations
def are_points_coplanar_torch(A, B, C, D):
    AB = B - A
    AC = C - A
    AD = D - A

    matrix = torch.stack([AB, AC, AD])  # 3x3 행렬

    determinant = torch.det(matrix)
    return torch.isclose(determinant, torch.tensor(0.0))

def postprocess(graph, vertices, step=0):
    faces = []
    for i, j in graph.edges:
        commons = sorted(list(nx.common_neighbors(graph, i, j)))
        for common in commons:
            if common<j: continue
            faces.append([i,j,common])

    if step==0:
        return faces

    removed_edge = []
    for face in faces:
        if (face[0],face[2]) in removed_edge:
            continue
        A = vertices[face[0]]
        B = vertices[face[1]]
        C = vertices[face[2]]
        
        AB = B - A
        AC = C - A
        
        cross_product = np.cross(AB, AC)
        area = np.linalg.norm(cross_product)
        
        if area==0:
            graph.remove_edge(face[0],face[2])
            removed_edge.append((face[0],face[2]))


    faces = []
    for i, j in graph.edges:
        commons = sorted(list(nx.common_neighbors(graph, i, j)))
        for common in commons:
            if common<j: continue
            faces.append([i,j,common])

    if step==1:
        return faces

    for _ in range(2):
        processed_square  = []
        removed_edge = set()
        saved_edge = set()
        n = 0
        while n < len(graph.edges):
            i,j = list(graph.edges)[n]
            n += 1
            if (i, j) in removed_edge: continue

            commons = sorted(list(nx.common_neighbors(graph, i, j)))
            break_sign = False
            for common in commons:
                clique = graph._adj[common].keys() & commons
                for comcom in clique:
                    square = sorted([i,j,common, comcom])
                    if not are_points_coplanar_torch(vertices[i], vertices[j], vertices[common], vertices[comcom]):
                        continue
                    if square in processed_square:
                        continue
                    total_edge = set(combinations(square, 2))
                    if total_edge & removed_edge:
                        continue

                    distance = torch.linalg.norm(vertices[square].unsqueeze(0) - vertices[square].unsqueeze(1),dim=-1)
                    max_value = torch.max(distance)
                    max_index = torch.argmax(distance)

                    row, col = divmod(max_index.item(), distance.size(1))
                    longest_edge = tuple(sorted((square[row], square[col])))
                    
                    edge = tuple(set(square) - set(longest_edge))
                    
                    if edge in saved_edge:
                        continue

                    graph.remove_edge(edge[0],edge[1])
                    processed_square.append(sorted(square))  
                    
                    saved_edge.add(edge)
                    removed_edge.add(edge)

                    for pair in total_edge:
                        saved_edge.add(pair)

                    if edge == (i,j):
                        break_sign = True
                    n -= 1
                    break
                if break_sign:
                    break
    
    faces = []
    for i, j in graph.edges:
        commons = sorted(list(nx.common_neighbors(graph, i, j)))
        for common in commons:
            if common<j: continue
            faces.append([i,j,common])

    return faces

from plyfile import PlyData, PlyElement
def save_point_cloud(pc, path='test_out.ply'):
    pcd = pc.cpu().detach().numpy()[:, :3]
    pcd = pcd / np.abs(pcd).max() * 0.50
    vertex = np.array([tuple(point) for point in pcd], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el], text=True).write(path)

