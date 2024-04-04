import torch
import json
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
import numpy as np
from utils.rotation_utils import matrix_to_rot6d
from utils.reconstruction import simple_recon_person
import argparse
import os
from os.path import join
from tqdm.auto import tqdm
from model.object_pose_fitting_batch import ObjectPoseFittingBatch, search_initial
import neural_renderer as nr
import cv2

# Check for CUDA availability and set the device accordingly
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

# Parse command-line arguments for setting up paths and configurations
parser = argparse.ArgumentParser(description='MPMO command line tools for 3D object tracking and reconstruction')
parser.add_argument('--seq_name', type=str, default="20230912/data01", help="Sequence name for processing")
parser.add_argument('--start_frame', type=int, default=0, help="Frame to start processing from")
parser.add_argument('--root_path', type=str, default='/nas/nas_10/AI-being/ZJY/', help="Root path for data storage")
parser.add_argument('--background_path', type=str, default='1,2,4,5,6', help="Indices for background images")
parser.add_argument('--foreground_path', type=str, default='3', help="Index for the foreground object")
parser.add_argument('--initial_path', type=str, help="Path for initial object position and rotation data")
parser.add_argument('--initial_search', action='store_true',
                    help="Flag to enable initial search for object positioning")
parser.add_argument('--step', type=int, default=60, help="Step size for frame iteration")
args = parser.parse_args()

# Define paths based on input arguments
root_path = join(args.root_path, args.seq_name)
object_idx = int(args.foreground_path)
seq_step = args.step

# Load object names and select the current object
txt_name = join(root_path, 'object_id.txt')
with open(txt_name, encoding='utf-8') as file:
    content = file.read().rstrip()
objects_name = content.split('\n')
object_name = objects_name[object_idx - 1].split(' ')[1]
object_path = join(args.root_path, 'Objects_template', object_name, object_name + '_simplified_transformed.obj')
output_path = join(root_path, 'output', 'object_1fps', object_name)

# Load camera calibration parameters
camera_path = join(root_path, 'calibration.json')
with open(camera_path, 'rb') as f:
    camera_param = json.load(f)

# Load and preprocess the 3D object mesh
verts, faces_idx, _ = load_obj(object_path)
verts = verts - verts.mean(0)  # Centering the vertices

# Load IMU data for tracking
imu_data_path = join(root_path, 'imu_data', object_name + '.json')
with open(imu_data_path, 'rb') as f:
    imu_data = json.load(f)

imu_startframe = int(imu_data['start_frame'])
rgb_startframe = imu_startframe
rgb_endframe = int(imu_data['end_frame'])

# Initialize reference transformations for tracking
object_T_reference = None

# Collect camera indices for processing
cams = []
if os.path.exists(join(root_path, 'videos')):
    cams_old = sorted(os.listdir(join(root_path, 'videos')))
    cams = [cam.split('.')[0] for cam in cams_old if cam[0] != '.']
    if cams[0].isdigit():
        cams = sorted(cams, key=lambda x: int(x))

# Ensure output directories exist
save_obj_dir = join(output_path, 'obj')
save_json_dir = join(output_path, 'json')
os.makedirs(save_obj_dir, exist_ok=True)
os.makedirs(save_json_dir, exist_ok=True)

num_view = len(cams)

if args.start_frame > 0:
    rgb_startframe = args.start_frame

# Main processing loop for frames
for frame in range(rgb_startframe, rgb_endframe):
    # Preprocess IMU data for current frame to obtain rotation
    object_R_IMU = np.array(imu_data['object_R'][frame - imu_startframe]).T.reshape(3, 3)

    # Skip frames based on the step size or if it's before the start frame
    if frame % seq_step != 0 and frame > rgb_startframe:
        continue

    # [Additional processing involving IMU data and condition to skip certain frames based on rotation offset]

    # Load and process masks for the current frame
    masks, _, masks_center, masks_background = load_mask_mpmo(args, root_path, cams, frame, object_idx)
    maskNs, maskNs_background, K_subs, R_subs, T_subs, P, object_masks_valid, object_keypoints = mask_cam_zoom_in(
        num_view, camera_param, masks, masks_background, masks_center)

    # Reconstruct 3D keypoints from 2D observations
    object_keypoints = np.stack(object_keypoints)
    keypoints3d, _ = simple_recon_person(object_keypoints, np.stack(P))

    # Initialize object transformation
    if args.initial_search == True or object_T_reference is None:
        object_init_T = keypoints3d[:, :3]
        object_init_R = matrix_to_rot6d(torch.from_numpy(object_R_IMU.reshape(1, 3, 3)).to(device))
    else:
        # Fallback or alternative initialization can be specified here
        pass

    # Setup and extend the 3D mesh with textures
    textures = TexturesVertex(verts_features=torch.ones_like(verts)[None].to(device))
    object_mesh = Meshes(verts=[verts.to(device)], faces=[faces_idx.verts_idx.to(device)], textures=textures).extend(
        num_view)

    # Initialize the neural renderer and the object pose fitting model
    silhouette_renderer = nr.renderer.Renderer(image_size=256, K=K_subs, R=R_subs, t=T_subs, orig_size=256,
                                               anti_aliasing=False)
    model = ObjectPoseFittingBatch(meshes=object_mesh, renderers=silhouette_renderer,
                                   image_refs=np.stack(maskNs).astype('float32'),
                                   Rs=torch.from_numpy(R_subs.transpose(0, 2, 1)), Ts=torch.from_numpy(T_subs[:, 0]),
                                   Ks=K_subs, object_masks_valid=object_masks_valid,
                                   image_refs_background=np.stack(maskNs_background).astype('float32'),
                                   object_R=object_init_R, object_init_T=object_init_T).to(device)

    # Optimization loop to fit the 3D model to observed data
    num_iter = 600 if object_T_reference is not None else 2000
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loop = tqdm(total=num_iter, desc="Optimizing model")
    for i in range(num_iter):
        optimizer.zero_grad()
        loss, _ = model()
        loss.backward()
        optimizer.step()

        # [Conditional logic to adjust model parameters and optimizer during the loop]

        loop.set_description(f"loss: {loss.data.item():.6g}")
        loop.update()
    loop.close()

    # Update and save the transformation for the current frame
    object_T_reference = model.translations.detach().cpu().numpy()
    object_R_reference = (model.object_R_IMU + model.rotations).detach().cpu()

    object_RT = {'object_T': model.translations.detach().cpu().numpy().tolist(),
                 'object_R': (model.object_R_IMU + model.rotations).detach().cpu().numpy().tolist()}
    if frame % seq_step == 0:
        with open(join(save_json_dir, f'{str(frame).zfill(6)}.json'), 'w') as json_file:
            json.dump(object_RT, json_file)

        mesh_temp = model.apply_transformation(model.rotations, model.translations)
        save_object_name = join(save_obj_dir, f'{str(frame).zfill(6)}.obj')
        save_obj(save_object_name, verts=mesh_temp[0], faces=model.faces[0])

        print(f"Processed frame {frame}")

# [Additional visualization or processing can be inserted here]
