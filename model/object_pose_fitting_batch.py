import torch
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import json
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    FoVPerspectiveCameras, look_at_view_transform, look_at_rotation,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, TexturesVertex,
)
import numpy as np
from utils.rotation_utils import rot6d_to_matrix, matrix_to_rot6d, make_rotate, points2mask, image_grid, \
    axis_angle_to_quaternion, quaternion_to_matrix, axis_angle_to_matrix, OpenGLRealPerspectiveCameras, \
    matrix_to_quaternion, quaternion_to_axis_angle
import os.path as osp
import cv2
# import neural_renderer as nr
from utils.reconstruction import simple_recon_person
from scipy.ndimage.morphology import distance_transform_edt
import argparse
import os
from os.path import join
from tqdm.auto import tqdm


class ObjectPoseFittingBatch(nn.Module):
    def __init__(self, meshes, renderers, image_refs, Rs, Ts, Ks, object_masks_valid, image_refs_background, object_R,
                 object_init_T):
        super().__init__()
        self.meshes = meshes
        self.device = meshes.device
        self.renderers = renderers
        verts = self.meshes.verts_padded()
        faces = self.meshes.faces_padded()
        # verts = self.meshes[0].verts_packed()
        # faces = self.meshes[0].faces_packed()
        self.register_buffer('verts', verts)
        self.register_buffer('faces', faces)
        self.register_buffer('Rs', Rs.to(meshes.device))
        self.register_buffer('Ts', Ts.to(meshes.device))
        Ks = np.stack(Ks)
        Ks = torch.from_numpy(Ks)
        self.register_buffer('Ks', Ks.to(meshes.device))
        image_refs = torch.from_numpy(image_refs).to(meshes.device)
        self.register_buffer('image_refs', image_refs)
        object_masks_valid = torch.from_numpy(object_masks_valid).to(meshes.device)
        self.register_buffer('object_masks_valid', object_masks_valid)
        image_refs_background = torch.from_numpy(image_refs_background).to(meshes.device)
        self.register_buffer('image_refs_background', image_refs_background)
        self.register_buffer('object_R_IMU', object_R.float())
        self.pool = torch.nn.MaxPool2d(
            kernel_size=7, stride=1, padding=(7 // 2)
        )
        mask_edge = self.compute_edges(image_refs).cpu().numpy()
        edt = np.array(
            [distance_transform_edt(1 - (mask_edge[i] > 0)) ** (0.25 * 2) for i in range(len(object_masks_valid))])
        self.register_buffer(
            "edt_ref_edge", torch.from_numpy(edt).float()
        )
        # self.register_buffer('edt_ref_edge', edt_ref_edge)
        rotation_init = torch.zeros(1, 3, 2)
        self.object_init_T = torch.from_numpy(object_init_T).to(meshes.device)
        translation_init = torch.from_numpy(object_init_T).to(meshes.device)
        self.rotations = nn.Parameter(rotation_init.clone().float().to(meshes.device), requires_grad=False)
        self.translations = nn.Parameter(translation_init.clone().float(), requires_grad=True)
        self.criterion_silhouette = nn.MSELoss()


    def apply_transformation(self, rotations, translations):

        rots = rot6d_to_matrix(self.object_R_IMU + 1. * rotations)
        verts_temp = torch.matmul(self.verts, rots) + translations

        return verts_temp

    def compute_edges(self, silhouette):
        return self.pool(silhouette) - silhouette

    def projection(self, vertices, K, R, t, dist_coeffs, orig_size, eps=1e-9):
        '''
        Calculate projective transformation of vertices given a projection matrix
        Input parameters:
        K: batch_size * 3 * 3 intrinsic camera matrix
        R, t: batch_size * 3 * 3, batch_size * 1 * 3 extrinsic calibration parameters
        dist_coeffs: vector of distortion coefficients
        orig_size: original size of image captured by the camera
        Returns: For each point [X,Y,Z] in world coordinates [u,v,z] where u,v are the coordinates of the projection in
        pixels and z is the depth
        '''

        # instead of P*x we compute x'*P'
        vertices = torch.matmul(vertices, R.transpose(2, 1)) + t.unsqueeze(1)
        x, y, z = vertices[:, :, 0], vertices[:, :, 1], vertices[:, :, 2]
        x_ = x / (z + eps)
        y_ = y / (z + eps)

        # Get distortion coefficients from vector
        k1 = dist_coeffs[:, None, 0]
        k2 = dist_coeffs[:, None, 1]
        p1 = dist_coeffs[:, None, 2]
        p2 = dist_coeffs[:, None, 3]
        k3 = dist_coeffs[:, None, 4]

        # we use x_ for x' and x__ for x'' etc.
        r = torch.sqrt(x_ ** 2 + y_ ** 2)
        x__ = x_ * (1 + k1 * (r ** 2) + k2 * (r ** 4) + k3 * (r ** 6)) + 2 * p1 * x_ * y_ + p2 * (r ** 2 + 2 * x_ ** 2)
        y__ = y_ * (1 + k1 * (r ** 2) + k2 * (r ** 4) + k3 * (r ** 6)) + p1 * (r ** 2 + 2 * y_ ** 2) + 2 * p2 * x_ * y_
        vertices = torch.stack([x__, y__, torch.ones_like(z)], dim=-1)
        vertices = torch.matmul(vertices, K.transpose(1, 2))
        u, v = vertices[:, :, 0], vertices[:, :, 1]
        v = orig_size - v
        # map u,v from [0, img_size] to [-1, 1] to use by the renderer
        u = 2 * (u - orig_size / 2.) / orig_size
        v = 2 * (v - orig_size / 2.) / orig_size
        vertices = torch.stack([u, v, z], dim=-1)
        return vertices

    def compute_offscreen_loss(self, verts, R, T, K):
        """
        Computes loss for offscreen penalty. This is used to prevent the degenerate
        solution of moving the object offscreen to minimize the chamfer loss.
        """
        # On-screen means xy between [-1, 1] and far > depth > 0
        proj = self.projection(
            verts,
            K,
            R.transpose(2, 1),
            T,
            torch.cuda.FloatTensor([[0., 0., 0., 0., 0.]]).to(self.device),
            orig_size=256,
        )
        xy, z = proj[:, :, :2], proj[:, :, 2:]
        zeros = torch.zeros_like(z)
        lower_right = torch.max(xy - 1, zeros).sum(dim=(1, 2))  # Amount greater than 1
        upper_left = torch.max(-1 - xy, zeros).sum(dim=(1, 2))  # Amount less than -1
        behind = torch.max(-z, zeros).sum(dim=(1, 2))
        too_far = torch.max(z - 200, zeros).sum(dim=(1, 2))
        return lower_right + upper_left + behind + too_far

    def forward(self):

        meshes_world = self.apply_transformation(self.rotations, self.translations)

        images = []
        R = self.Rs
        T = self.Ts
        K = self.Ks

        image = (1 - self.image_refs_background) * self.renderers(meshes_world, self.faces, mode="silhouettes")
        # image = self.renderers(meshes_world, self.faces, mode="silhouettes")
        loss = torch.sum((image - self.image_refs) ** 2, dim=(1, 2)) * self.object_masks_valid
        loss_offscreen = self.compute_offscreen_loss(meshes_world, R.float(), T.float(), K.float()) * self.object_masks_valid
        loss_sum = torch.sum((1. * loss + 10. * loss_offscreen) / self.object_masks_valid.sum())
        images.append(image)

        return loss_sum , images

    def render_only(self,rotations, translations):

        meshes_world = self.apply_transformation(rotations, translations)

        images = []
        num_view = len(self.object_masks_valid)
        R = self.Rs
        T = self.Ts
        K = self.Ks

        # image =  (1 - self.image_refs_background) * self.renderers(meshes_world=meshes_world, R=R, T=T)[..., 3]
        image =  (1 - self.image_refs_background) * self.renderers(meshes_world, self.faces, mode="silhouettes")

        loss = torch.sum((image - self.image_refs) ** 2, dim=(1, 2)) * self.object_masks_valid
        loss_offscreen = self.compute_offscreen_loss(meshes_world, R.float(), T.float(), K.float()) * self.object_masks_valid
        # loss_sum = torch.sum((1. * loss + 0 * loss_offscreen) / self.object_masks_valid.sum())
        # loss_sum = ( torch.sum(loss) + 1. * torch.sum(loss_offscreen) ) /  self.object_masks_valid.sum()
        # loss_reg = torch.sum(self.rotations ** 2)
        # loss_sum = torch.sum((1. * loss + 10. * loss_offscreen) / self.object_masks_valid.sum())
        loss_sum = torch.sum((1. * loss) / self.object_masks_valid.sum())

        images.append(image)

        return loss_sum , images


def search_initial(model,image_refs):

    loss_min = 1e10
    rotations_best = torch.zeros(1, 3, 2)
    # for i in tqdm(range(10000)):
    for i in range(10000):
        translations = model.translations
        rotations = 100. * torch.rand(1, 3, 2).to(model.device)
        loss , images = model.render_only(rotations,translations)
        if loss < loss_min:
            loss_min = loss
            rotations_best = rotations.clone()
    return rotations_best