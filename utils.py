import os, math, time, json
import numpy as np
from collections import deque
from typing import List
import torch
import torch.distributed as dist

from plyfile import PlyData, PlyElement
import trimesh

from PIL import Image

def make_gif_from_images(frames, output_gif, duration=30):
    # Save frames as a GIF
    frames[0].save(output_gif, format="GIF", append_images=frames[1:], 
                   save_all=True, duration=duration, loop=0)


def process_mesh_to_pc(mesh, sample_num = 4096):
    # mesh_list : list of trimesh
    pc_coor, face_idx = mesh.sample(sample_num, return_index=True)
    normals = mesh.face_normals[face_idx]

    normal_norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / normal_norm

    bounds = np.array([pc_coor.min(axis=0), pc_coor.max(axis=0)])
    pc_coor = pc_coor - (bounds[0] + bounds[1])[None, :] / 2
    pc_coor = pc_coor / np.abs(pc_coor).max() * 0.99
    
    if not (-0.99 <= pc_coor).all() or not (pc_coor <= 0.99).all(): None

    pc_normal = np.concatenate([pc_coor, normals], axis=-1, dtype=np.float16)
    return pc_normal

def save_pc(pcd, save_path):
    pcd = pcd / np.abs(pcd).max() * 0.50
    vertex = np.array([tuple(point) for point in pcd], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el], text=True).write(save_path)

def make_faces_to_mesh(recon_mesh):
    valid_mask = torch.all(~torch.isnan(recon_mesh.reshape((-1,9))), dim=1)
    recon_mesh = recon_mesh[valid_mask]  # nvalid_face x 3 x 3
    vertices = recon_mesh.reshape(-1, 3).cpu()
    vertices_index = np.arange(len(vertices))  # 0, 1, ..., 3 x face
    triangles = vertices_index.reshape(-1, 3)

    scene_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, force="mesh",
                                    merge_primitives=True)
    scene_mesh.merge_vertices()
    scene_mesh.update_faces(scene_mesh.nondegenerate_faces())
    scene_mesh.update_faces(scene_mesh.unique_faces())
    scene_mesh.remove_unreferenced_vertices()
    scene_mesh.fix_normals()

    return scene_mesh

@torch.jit.ignore
def to_list_1d(arr) -> List[float]:
    arr = arr.detach().cpu().numpy().tolist()
    return arr


@torch.jit.ignore
def to_list_3d(arr) -> List[List[List[float]]]:
    arr = arr.detach().cpu().numpy().tolist()
    return arr


def huber_loss(error, delta=1.0):
    """
    Ref: https://github.com/charlesq34/frustum-pointnets/blob/master/models/model_util.py
    x = error = pred - gt or dist(pred,gt)
    0.5 * |x|^2                 if |x|<=d
    0.5 * d^2 + d * (|x|-d)     if |x|>d
    """
    abs_error = torch.abs(error)
    quadratic = torch.clamp(abs_error, max=delta)
    linear = abs_error - quadratic
    loss = 0.5 * quadratic ** 2 + delta * linear
    return loss

def save_checkpoint(
    checkpoint_dir,
    model,
    optimizer,
    epoch,
    args,
    best_val_metrics,
    filename=None,
):

    checkpoint_name = os.path.join(checkpoint_dir, filename)

    # model = model.half()
    try:
        weight_ckpt = model.module.state_dict()
    except Exception as e:
        print("single GPU")
        weight_ckpt = model.state_dict()

    sd = {
        "model": weight_ckpt,
        "epoch": epoch,
        "args": args,
        "best_val_metrics": best_val_metrics,
    }
    torch.save(sd, checkpoint_name)

def compute_learning_rate(args, curr_epoch_normalized):
    assert curr_epoch_normalized <= 1.0 and curr_epoch_normalized >= 0.0
    if (
        curr_epoch_normalized <= (args.warm_lr_epochs / args.max_epoch)
        and args.warm_lr_epochs > 0
    ):
        # Linear Warmup
        curr_lr = args.warm_lr + curr_epoch_normalized * args.max_epoch * (
            (args.base_lr - args.warm_lr) / args.warm_lr_epochs
        )
    else:
        # Cosine Learning Rate Schedule
        curr_lr = args.final_lr + 0.5 * (args.base_lr - args.final_lr) * (
            1 + math.cos(math.pi * curr_epoch_normalized)
        )
    return curr_lr

def adjust_learning_rate(args, optimizer, curr_epoch):
    curr_lr = compute_learning_rate(args, curr_epoch)
    for param_group in optimizer.param_groups:
        param_group["lr"] = curr_lr
    return curr_lr


def all_reduce_sum(tensor):
    if not is_distributed():
        return tensor
    dim_squeeze = False
    if tensor.ndim == 0:
        tensor = tensor[None, ...]
        dim_squeeze = True
    torch.distributed.all_reduce(tensor)
    print("loss_tensor: ", tensor)
    if dim_squeeze:
        tensor = tensor.squeeze(0)
    return tensor

def is_distributed():
    if not dist.is_available() or not dist.is_initialized():
        return False
    return True

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_distributed():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        torch.distributed.barrier()
        all_reduce_sum(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


def copy_state_dict(cur_state_dict, pre_state_dict, prefix = '', drop_prefix='', fix_loaded=False):
    success_layers, failed_layers = [], []
    def _get_params(key):
        key = key.replace(drop_prefix,'')
        key = prefix + key
        if key in pre_state_dict:
            return pre_state_dict[key]
        return None

    for k in cur_state_dict.keys():
        v = _get_params(k)
        try:
            if v is None:
                failed_layers.append(k)
                continue
            cur_state_dict[k].copy_(v)
            if prefix in k and prefix!='':
                k=k.split(prefix)[1]
            success_layers.append(k)
        except:
            print('copy param {} failed, mismatched'.format(k)) # logging.info
            continue
    print('missing parameters of layers:{}'.format(failed_layers))

    if fix_loaded and len(failed_layers)>0:
        print('fixing the layers that were loaded successfully, while train the layers that failed,')
        for k in cur_state_dict.keys():
            try:
                if k in success_layers:
                    cur_state_dict[k].requires_grad=False
            except:
                print('fixing the layer {} failed'.format(k))

    return success_layers


def load_model(pretrained_model, model, prefix = '', drop_prefix='',optimizer=None, **kwargs):

    # pretrained_model = torch.load(path)
    current_model = model.state_dict()
    if isinstance(pretrained_model, dict):
        if 'model_state_dict' in pretrained_model:
            pretrained_model = pretrained_model['model_state_dict']
    copy_state_dict(current_model, pretrained_model, prefix = prefix, drop_prefix=drop_prefix, **kwargs)

    return model


from einops import rearrange

def loop_detokenize(input_ids):
        input_ids = input_ids.reshape(input_ids.shape[0], -1) # B x L
        batch_size = input_ids.shape[0]
        continuous_coors = torch.zeros((batch_size, 3000, 3), device=input_ids.device)
        continuous_coors[...] = float('nan')
        for i in range(batch_size):
            cur_ids = input_ids[i]
            coor_loop_check = 0
            vertice_count = 0
            continuous_coors[i, :3, :] = torch.tensor([[-0.1, 0.0, 0.1], [-0.1, 0.1, 0.2], [-0.3, 0.3, 0.2]],
                                                      device=input_ids.device)
            error_judge = 0
            for id in cur_ids:
                if id == -1:
                    if coor_loop_check < 9:
                        error_judge=1
                    if coor_loop_check % 3 != 0:
                        error_judge=1
                    break
                elif id == 128:
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
                    continuous_coors[i, vertice_count, coor_loop_check % 3] = undiscretize(id, -0.5, 0.5, 128)
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
