import torch
import torch.nn as nn
import copy
import numpy as np
import time
from scipy.spatial import KDTree
from utils import load_model
from models.models import register, MODELS
from huggingface_hub import PyTorchModelHubMixin

def compute_vertex_normals(point_cloud, vertices, k=10):
    vertices = (vertices-vertices.min())/(vertices.max()-vertices.min())*2 - 1

    # Separate coordinates and normals from the point cloud
    coords = point_cloud[:, :3]
    normals = point_cloud[:, 3:]
    
    # Build KDTree for nearest neighbor search
    tree = KDTree(coords)
    vertex_normals = np.zeros((vertices.shape[0], 3))
    
    for i, v in enumerate(vertices):
        # Query the k nearest neighbors of vertex v
        _, indices = tree.query(v, k=k)
        
        # Average the normals of the neighbors
        neighbor_normals = normals[indices]
        avg_normal = np.mean(neighbor_normals, axis=0)
        
        # Normalize the resulting normal vector
        norm = np.linalg.norm(avg_normal) + 1e-8  # Avoid division by zero
        vertex_normals[i] = avg_normal / norm
    
    return vertex_normals

@register("MeshGen")
class MeshGen(nn.Module, PyTorchModelHubMixin):
    def __init__(self, **kwargs):
        super().__init__()

        self.vert_args = kwargs.copy()
        self.vert_args['n_discrete_size'] = 128
        self.vert_gen = MODELS[self.vert_args['vert_model']](self.vert_args)

        self.face_args = kwargs.copy()
        self.face_args['n_discrete_size'] = 512
        self.face_gen = MODELS[self.face_args['face_model']](self.face_args)


        if self.vert_args['vertgen_ckpt'] is not None:
            load_model(torch.load(self.vert_args['vertgen_ckpt'], map_location=torch.device("cpu"), weights_only=False)["model"], self.vert_gen)
        if self.face_args['vertgen_ckpt'] is not None:
            load_model(torch.load(self.face_args['facegen_ckpt'], map_location=torch.device("cpu"), weights_only=False)["model"], self.face_gen)

    def forward(self, data_dict: dict, is_eval: bool=True) -> dict:
        vertices, _  = self.vert_gen(data_dict, is_eval=is_eval)
        sequence = vertices.clone().long()
        sequence[sequence!=-1] = ((vertices[vertices!=-1]+0.5)*self.face_args['n_discrete_size']).long()

        data_dict['vertices'] = vertices
        data_dict['sequence'] = sequence
        gen_mesh = self.face_gen(data_dict, is_eval=is_eval)
        
        return gen_mesh    