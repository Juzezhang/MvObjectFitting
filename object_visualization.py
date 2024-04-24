import sys
import os
from os.path import join
import json
import cv2
import numpy as np
import torch
import argparse
from tqdm import tqdm
# Importing custom modules for 3D visualization and SMPL model handling
from viz.pyt3d_wrapper import Pyt3DWrapper
from psbody.mesh import Mesh
from model.body_models_easymocap.smpl import SMPLModel
from yacs.config import CfgNode
from utils.rotation_utils import rot6d_to_matrix
import matplotlib.pyplot as plt
import trimesh

# Set up the computation device based on CUDA availability
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

# Parse command line arguments
parser = argparse.ArgumentParser(description='hoim3 command line tools')
parser.add_argument('--root_path', type=str, default="/nas/nas_38/AI-being/elan/20240328/basketball_demo")
parser.add_argument('--resolution', type=int, default=1080)
parser.add_argument('--step', type=int, default=60)
parser.add_argument('--output_path', type=str, default="/nas/nas_10/AI-being/20240328/basketball_demo/output/vis")
args = parser.parse_args()

# Setup paths
root_path = args.root_path
mocap_path = join(args.root_path, 'output')
seq_step = args.step
body_model = SMPLModel(model_path='model/model_files/smpl/SMPL_NEUTRAL.pkl',device=device)

# Load object template
object_names = os.listdir(join(mocap_path,'object'))
object_mesh_list = []
object_vertex_list = []
for object_name in object_names:
    object_template_path = join(root_path, 'ObjectTemplate', object_name + '.obj')
    object_mesh = trimesh.load_mesh(object_template_path)
    object_mesh.vertices -= object_mesh.vertices.mean(0)
    # object_mesh.v -= object_mesh.v.mean(0)
    object_mesh_psbody = Mesh(f=object_mesh.faces)
    object_mesh_psbody.v = object_mesh.vertices
    object_mesh_list.append(object_mesh_psbody)
    object_vertex_list.append(object_mesh_psbody.v)
    # object_mesh_vertices = object_mesh.v

# Load calibration data
calibration_path = join(root_path, 'calibration.json')

# Load camera parameters
with open(calibration_path, 'rb') as file:
    camera_params = json.load(file)


imu_data_path = join(root_path, 'imu_data', object_names[0] + '.json')
# with open(imu_data_path, 'rb') as f:
#     imu_data = json.load(f)
# start_frame = int(imu_data['start_frame'])
# end_frame = int(imu_data['end_frame'])

videos_path = join(root_path, 'videos')
video_files = sorted(os.listdir(videos_path))

# Process each video
for video_name in video_files:
    view_index = int(video_name.split('.')[0])
    output_dir = join(args.output_path, str(view_index))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f'Rendering image with camera view {view_index}')
    scaling_factor = 2160 / args.resolution
    # Intrinsic camera matrix adjustment
    K = np.array(camera_params[str(view_index)]['K']).reshape((3, 3))
    K /= scaling_factor
    K[2, 2] = 1
    # Extrinsic camera parameters
    RT = np.array(camera_params[str(view_index)]['RT'])
    if len(RT) == 16:
        RT = RT.reshape((4, 4))
    else:
        RT = RT.reshape((3, 4))

    R = RT[:3, :3]
    T = RT[:3, 3:].reshape(3)
    # Process video frames
    video_path = join(videos_path, video_name)
    capture = cv2.VideoCapture(video_path)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    # start_frame = start_frames['rgb'][args.seq_name]
    for frame_count in tqdm(range(0, total_frames), desc="Processing frames"):
        if frame_count % seq_step != 0:
            continue
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        _, img = capture.read()
        img = cv2.resize(img[:, :, ::-1], (int(3840 / scaling_factor), int(2160 / scaling_factor)))
        # # Load SMPL parameters for the current frame
        # smpl_param_path = join(mocap_path, 'smpl_autotrack', f'{str(frame_count).zfill(6)}.json')
        # with open(smpl_param_path, 'rb') as file:
        #     smpl_params = json.load(file)
        # # Convert SMPL parameters to NumPy arrays
        # smpl_params_list = []
        # for smpl_param in smpl_params:
        #     smpl_params_list.append({key: np.array(value) if isinstance(value, list) else value for key, value in smpl_param.items()})
        # # Update human mesh vertices
        # out_mesh_list = []
        # for smpl_param in smpl_params_list:
        #     out_mesh_list.append(body_model(**smpl_param))
        #
        # human_mesh_list = []
        # for out_mesh in out_mesh_list:
        #     human_mesh = Mesh(f=body_model.faces)
        #     human_mesh.v = out_mesh[0].cpu().numpy()
        #     human_mesh_list.append(human_mesh)

        # Object transformation
        for object_idx, object_name in enumerate(object_names):
            object_RT_path = join(mocap_path, 'object', object_name, 'json', f'{str(frame_count).zfill(6)}.json')
            with open(object_RT_path, 'rb') as file:
                object_rotation_refinement = json.load(file)
            object_R_refinement = np.array(object_rotation_refinement['object_R'])
            object_R_refinement = torch.from_numpy(object_R_refinement)
            object_R_refinement = rot6d_to_matrix(object_R_refinement).numpy().reshape(3, 3).T
            object_T_refinement = np.array(object_rotation_refinement['object_T']).reshape(1, 3)
            # Apply transformation to the object mesh vertices
            object_mesh_vertices = object_vertex_list[object_idx]
            object_mesh_list[object_idx].v = object_mesh_vertices.dot(object_R_refinement.T) + object_T_refinement


        # Initialize 3D visualization wrapper
        pyt3d_wrapper = Pyt3DWrapper(image_size=(int(args.resolution), int(args.resolution*3840/2160)), K=K, R=R, T=T, image=img)
        # Render meshes
        # rendered_image = pyt3d_wrapper.render_meshes(human_mesh_list + object_mesh_list, False, R=R, T=T)
        rendered_image = pyt3d_wrapper.render_meshes(object_mesh_list, False, R=R, T=T)
        # Save the rendered image
        save_path = join(output_dir, f'{str(frame_count).zfill(6)}.jpg')
        cv2.imwrite(save_path, 255 * rendered_image[:, :, ::-1])
        # Optional: display the rendered image
        # plt.imshow(rendered_image)
        # plt.show()
    capture.release()
