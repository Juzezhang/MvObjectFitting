# Import necessary libraries
import torch
import json
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
import numpy as np
from utils.rotation_utils import rot6d_to_matrix, matrix_to_rot6d, image_grid
from utils.reconstruction import simple_recon_person
import argparse
import os
from os.path import join
from tqdm.auto import tqdm
from model.object_pose_fitting_batch import ObjectPoseFittingBatch, search_initial
import neural_renderer as nr
from utils.vis import show
from utils.loading_file import load_mask_mpmo, mask_cam_zoom_in
import cv2
import matplotlib.pyplot as plt

# Set the CUDA device for computation
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

# Parse command line arguments for configuration
parser = argparse.ArgumentParser(description='SPSO command line tools')
parser.add_argument('--start_frame', type=int, default=0, help='Starting frame for processing')
parser.add_argument('--root_path', type=str, default='/nas/nas_10/AI-being/20240328/basketball_demo', help='Root directory for input data')
parser.add_argument('--background_path', type=str, default='2,3', help='Paths for background images')
parser.add_argument('--foreground_path', type=str, default='1', help='Path for foreground object')
parser.add_argument('--initial_search', action='store_true', help='Flag to perform initial search')
parser.add_argument('--optimize_R', action='store_true', help='Flag to optimize rotation')
parser.add_argument('--step', type=int, default=1, help='Step size for frame processing')
args = parser.parse_args()

# Setup paths based on arguments
root_path = args.root_path
object_idx = int(args.foreground_path)
seq_step = args.step
txt_name = os.path.join(root_path, 'object_id.txt')

# Read object names from a file
with open(txt_name, encoding='utf-8') as file:
    content = file.read().rstrip()
objects_name = content.split('\n')
object_name = objects_name[object_idx-1].split(' ')[1]
object_path = join(args.root_path, 'ObjectTemplate', object_name + '.obj')
output_path = join(root_path, 'output', 'object', object_name)

# Load camera calibration parameters
camera_path = join(root_path, 'calibration.json')
with open(camera_path, 'rb') as f:
    camera_param = json.load(f)

# Load the object mesh
verts, faces_idx, _ = load_obj(object_path)
verts = verts - verts.mean(0)

# Set up video capture to determine the number of frames
video_path = join(root_path, 'videos/0.mp4')
capture = cv2.VideoCapture(video_path)
rgb_endframe = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

object_T_reference = None

# Load camera indices if videos are available
cams = []
if os.path.exists(join(root_path, 'videos')):
    cams_old = sorted(os.listdir(join(root_path, 'videos')))
    cams = [cam.split('.')[0] for cam in cams_old if cam[0] != '.']
    if cams[0].isdigit():
        cams = sorted(cams, key=lambda x: int(x))

# Create directories for saving results if they do not exist
save_obj_dir = join(output_path, 'obj')
save_json_dir = join(output_path, 'json')
os.makedirs(save_obj_dir, exist_ok=True)
os.makedirs(save_json_dir, exist_ok=True)

num_view = len(cams)
if args.start_frame > 0:
    rgb_startframe = args.start_frame
else:
    rgb_startframe = 0

# Main loop for processing each frame
for frame in range(rgb_startframe, rgb_endframe):
    if frame % seq_step != 0 and frame > rgb_startframe:
        continue

    # Load masks and camera parameters for the current frame
    masks, omask, masks_center, masks_background = load_mask_mpmo(args, root_path, cams, frame, object_idx)
    maskNs, maskNs_background, K_subs, R_subs, T_subs, P, object_masks_valid, object_keypoints = mask_cam_zoom_in(num_view, camera_param, masks, masks_background, masks_center)

    Pall = np.stack(P)
    object_keypoints = np.stack(object_keypoints)
    # Reconstruct 3D keypoints from 2D observations
    keypoints3d, kpts_repro = simple_recon_person(object_keypoints, Pall)

    image_refs = np.stack(maskNs).astype('float32')
    image_refs_background = np.stack(maskNs_background).astype('float32')
    ############ mask display ############
    # t1 = np.stack(maskNs)
    # t2 = np.stack(masks)
    # plt.rcParams['figure.dpi'] = 120
    # col_num = 5
    # image_grid(t1, rows=len(maskNs)//col_num, cols=col_num ,rgb=False)
    # image_grid(t2, rows=len(masks)//col_num, cols=col_num ,rgb=False)
    ############ mask display ############

    # Initialize object transformation
    if object_T_reference is None:
        object_init_T = keypoints3d[:, :3]
        object_init_R = torch.eye(3).reshape(1, 3, 3)[:, :, :2].to(device)
    else:
        object_init_T = object_T_reference
        object_init_R = torch.eye(3).reshape(1, 3, 3)[:, :, :2].to(device)

    faces = faces_idx.verts_idx
    verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
    textures = TexturesVertex(verts_features=verts_rgb.to(device))
    # Create a Meshes object for the object with textures
    object_mesh = Meshes(
        verts=[verts.to(device)],
        faces=[faces.to(device)],
        textures=textures
    ).extend(num_view)

    # Setup the neural renderer
    silhouette_renderer = nr.renderer.Renderer(
        image_size=256,
        K=K_subs,
        R=R_subs,
        t=T_subs,
        orig_size=256,
        anti_aliasing=False,
    )

    Rs = torch.from_numpy(R_subs.transpose(0, 2, 1))
    Ts = torch.from_numpy(T_subs[:, 0])
    # Initialize the pose fitting model
    model = ObjectPoseFittingBatch(meshes=object_mesh, renderers=silhouette_renderer, image_refs=image_refs, Rs=Rs, Ts=Ts, Ks=K_subs, object_masks_valid=object_masks_valid, image_refs_background=image_refs_background, object_R=object_init_R, object_init_T=object_init_T).to(device)

    # Set the number of iterations and optimizer based on whether this is the initial frame
    if object_T_reference is not None:
        num_iter = 100
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        begin_iter = 0.1
    else:
        num_iter = 2000
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        begin_iter = 0.1

    # Optimization loop
    loss_pre = 0.
    loss = 0.
    loop = tqdm(total=num_iter, desc='Optimization progress')
    for i in range(num_iter):
        optimizer.zero_grad()
        loss_pre = loss
        loss, images = model()
        loss.backward()
        optimizer.step()
        loop.set_description(f"loss: {loss.data.item():.6g}")
        loop.update()
    loop.close()

    # Save the optimized pose and transformation for this frame
    object_T_reference = model.translations.detach().cpu().numpy()
    object_R_reference = (model.object_R_IMU + 1. * model.rotations).detach().cpu()

    object_T = model.translations.detach().cpu().numpy().tolist()
    object_R = (model.object_R_IMU + 1. * model.rotations).detach().cpu().numpy().tolist()

    object_RT = {'object_T': object_T, 'object_R': object_R}

    # Save results every sequence step
    if frame % seq_step == 0:
        with open(join(save_json_dir, f'{str(frame).zfill(6)}.json'), 'w') as json_file:
            json.dump(object_RT, json_file)

        mesh_temp = model.apply_transformation(model.rotations, model.translations)
        save_object_name = join(save_obj_dir, f'{str(frame).zfill(6)}.obj')
        save_obj(save_object_name, verts=mesh_temp[0], faces=model.faces[0])

        print(f'Frame {frame} processed')
