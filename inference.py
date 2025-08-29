import os
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.utils import DistributedDataParallelKwargs
import trimesh
import glob
import torch
import numpy as np

from main import make_args_parser
from models import MODELS

def apply_normalize(mesh):
    '''
    normalize mesh to [-1, 1]
    '''
    bbox = mesh.bounds
    center = (bbox[1] + bbox[0]) / 2
    scale = (bbox[1] - bbox[0]).max()

    mesh.apply_translation(-center)
    mesh.apply_scale(1 / scale * 2 * 0.95)

    return mesh

def sample_pc(mesh, pc_num, with_normal=False):
    mesh = apply_normalize(mesh)
    
    if not with_normal:
        points, _ = mesh.sample(pc_num, return_index=True)
        return points

    points, face_idx = mesh.sample(50000, return_index=True)
    normals = mesh.face_normals[face_idx]
    pc_normal = np.concatenate([points, normals], axis=-1, dtype=np.float16)

    # random sample point cloud
    ind = np.random.choice(pc_normal.shape[0], pc_num, replace=False)
    pc_normal = pc_normal[ind]
    
    return pc_normal

if __name__ == "__main__":
    args = make_args_parser()

    accelerator = Accelerator(
        mixed_precision=args.precision,
        log_with="wandb",
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)]
    )

    set_seed(args.seed, device_specific=True)
    if args.variant=="V1K":
        model = MODELS[args.model_name].from_pretrained("WopperSet/FastMesh-V1K")
    elif args.variant=="V4K":
        model = MODELS[args.model_name].from_pretrained("WopperSet/FastMesh-V4K")
    model = accelerator.prepare(model)
    model.eval()


    EXT = ["obj", "glb", "ply"]

    mesh_lists = []
    for ext in EXT:
        mesh_lists += glob.glob(args.mesh_path+f"/*.{ext}")

    batch_size = args.batch_size
    outputs = args.outputs
    os.makedirs(outputs, exist_ok=True)

    input_shape = torch.empty((0,args.input_pc_num,6)).cuda()
    mesh_paths = []
    for id, mesh_path in enumerate(mesh_lists):
        gt_mesh = apply_normalize(trimesh.load(mesh_path, force='mesh'))
        pc_normal = sample_pc(gt_mesh, args.input_pc_num, with_normal=True)
        pc_normal = torch.from_numpy(pc_normal).unsqueeze(0).cuda()
        input_shape = torch.cat([input_shape, pc_normal])
        mesh_paths.append(mesh_path)
        
        if input_shape.shape[0]<batch_size and id<len(mesh_lists)-1:
            continue
        
        input_dict = {"pc_normal": input_shape}

        with accelerator.autocast():
            recon_meshes = model(input_dict, is_eval=True)

        for path, mesh in zip(mesh_paths,recon_meshes):
            save_path = os.path.join(outputs, os.path.basename(path))[:-4]+f"_{args.variant}.obj"
            brown_color = np.array([255, 165, 0, 255], dtype=np.uint8)
            face_colors = np.tile(brown_color, (len(mesh.faces), 1))
            mesh.visual.face_colors = face_colors

            mesh.export(save_path)

        input_shape = torch.empty((0,args.input_pc_num,6)).cuda()
        mesh_paths = []

    print(f"GENERATE ALL MESHES AT {outputs} FOLDER")
