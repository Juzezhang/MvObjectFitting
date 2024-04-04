import torch
import json
from pytorch3d.io import load_obj , save_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import  TexturesVertex
import numpy as np
from utils.rotation_utils import rot6d_to_matrix,matrix_to_rot6d, image_grid
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

# Set the cuda device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


parser = argparse.ArgumentParser('MPMO command line tools')
parser.add_argument('--seq_name', type=str, default="20230912/data01")
parser.add_argument('--start_frame', type=int, default=0)
# parser.add_argument('--object_idx', type=int, default=3)
# parser.add_argument('--camera_path', type=str, default= '/nas/nas_10/NeuralDome/NeuralDome/rgb/20221018/calib/calibration.json')
parser.add_argument('--root_path', type=str, default='/nas/nas_10/AI-being/ZJY/')
parser.add_argument('--background_path', type=str, default='1,2,4,5,6')
parser.add_argument('--foreground_path', type=str, default='3')
parser.add_argument('--initial_path', type=str, default='/nas/nas_10/AI-being/ZJY/20230912/data01/result/smallsofa/json')
parser.add_argument('--initial_search', action='store_true')
parser.add_argument('--step', type=int, default=60)

args = parser.parse_args()
print(args.seq_name)
root_path = join(args.root_path, args.seq_name)
object_idx = int(args.foreground_path)
seq_step = args.step

txt_name = os.path.join(root_path, 'object_id.txt')
with open(txt_name, encoding='utf-8') as file:
    content = file.read()
    content = content.rstrip()
objects_name = content.split('\n')
object_name = objects_name[object_idx-1].split(' ')[1]
object_path = join(args.root_path, 'Objects_template', object_name, object_name + '_simplified_transformed.obj')
output_path = join(root_path, 'output', 'object_1fps',object_name)


camera_path = join(root_path,'calibration.json')
with open(camera_path, 'rb') as f:
    camera_param = json.load(f)

verts, faces_idx, _ = load_obj(object_path)
verts = verts - verts.mean(0)

imu_data_path = join(root_path, 'imu_data', object_name + '.json')
with open(imu_data_path, 'rb') as f:
    imu_data = json.load(f)


imu_startframe = int(imu_data['start_frame'])
rgb_startframe = imu_startframe

rgb_endframe = int(imu_data['end_frame'])
object_T_reference = None


cams = []
if len(cams) == 0 and os.path.exists(join(root_path, 'videos')):
    cams_old = sorted(os.listdir(join(root_path, 'videos')))
    cams = []
    for cam in cams_old:
        # if cam != '.DS_Store':
        if cam[0] != '.':
            cams.append(cam.split('.')[0])
    if cams[0].isdigit():
        cams = sorted(cams, key=lambda x: int(x))


save_obj_dir = join(output_path, 'obj')
save_json_dir = join(output_path, 'json')

if not os.path.exists(save_obj_dir):
    os.makedirs(save_obj_dir)
if not os.path.exists(save_json_dir):
    os.makedirs(save_json_dir)


num_view = len(cams)
object_R_IMU = 0
object_T_marker = 0
# rgb_startframe = 272
if args.start_frame > 0:
    rgb_startframe = args.start_frame

for frame in range(rgb_startframe, rgb_endframe):
    object_R_IMU_pre = object_R_IMU
    object_R_IMU = np.array(imu_data['object_R'][frame-imu_startframe]).T.reshape(3, 3)
    # object_R_IMU = torch.from_numpy(object_R_IMU).to(device)
    # object_R_IMU = matrix_to_rot6d(object_R_IMU)
    if frame % seq_step != 0 and frame > rgb_startframe:
        continue
    if frame > rgb_startframe:
        # acc = torch.from_numpy(np.array(imu_data['object_T'][:frame - imu_startframe]))
        # acc_next = torch.from_numpy(np.array(imu_data['object_T'][:frame - imu_startframe + 1]))
        # sampling_rate = 60
        # time_step = 1.0 / sampling_rate
        # velocities_per = cumtrapz(acc, dx=time_step, initial=0.0)[-1, :]
        # velocities = cumtrapz(acc_next, dx=time_step, initial=0.0)[-1, :]
        # offset_T = torch.from_numpy(velocities_per - velocities).abs().sum()

        # offset_R = (object_R_IMU - object_R_IMU_pre).abs().sum()
        R_rel = object_R_IMU_pre @ object_R_IMU.T
        trace_R_rel = np.trace(R_rel)
        # 确保角度计算的输入在[-1, 1]范围内
        cos_theta = (trace_R_rel - 1) / 2
        cos_theta_clipped = np.clip(cos_theta, -1, 1)
        # 计算角度距离
        theta = np.arccos(cos_theta_clipped)
        # 将角度转换为度数
        theta_degrees = np.degrees(theta)
        offset_R = theta_degrees
        # if offset_R < 0.1 and offset_T < 0.02:
        if offset_R < 0.05:
        # if (offset_R + 0.1* offset_T ) < 0.0001:
            object_T = model.translations.detach().cpu().numpy().tolist()
            object_R = (model.object_R_IMU + 1. * model.rotations).detach().cpu().numpy().tolist()
            object_RT = dict()
            object_RT['object_T'] = object_T
            object_RT['object_R'] = object_R
            with open(join(save_json_dir, '{}.json'.format(str(frame).zfill(6))), 'w') as json_file:
                json.dump(object_RT, json_file)

            mesh_temp = model.apply_transformation(model.rotations, model.translations)
            save_object_name = join(save_obj_dir, '{}.obj'.format(str(frame).zfill(6)))
            save_obj(save_object_name, verts=mesh_temp[0], faces=model.faces[0])
            print(frame)
            continue


    masks, omask, masks_center, masks_background = load_mask_mpmo(args, root_path, cams, frame,object_idx)

    maskNs, maskNs_background, K_subs, R_subs, T_subs, P, object_masks_valid, object_keypoints = mask_cam_zoom_in(num_view, camera_param, masks, masks_background, masks_center)

    Pall = np.stack(P)
    object_keypoints = np.stack(object_keypoints)
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

    if args.initial_search == True:
        object_init_T = keypoints3d[:, :3]
        object_init_R = matrix_to_rot6d(torch.from_numpy(object_R_IMU.reshape(1, 3, 3)).to(device))
    else:
        if object_T_reference is None:
            try:
                # initial_RT_path = os.path.join(args.initial_path, str(frame) + '.json')
                # with open(initial_RT_path, 'rb') as f:
                #     object_rotation = json.load(f)
                # object_R_intial = np.array(object_rotation['object_R']).reshape(1, 3, 2)
                # object_R_intial = torch.from_numpy(object_R_intial).to(device)
                # object_T_initial = np.array(object_rotation['object_T'])
                # object_init_T = object_T_initial
                # # object_init_T = keypoints3d[:, :3]
                # object_init_R = object_R_intial
                object_init_T = keypoints3d[:, :3]
                object_init_R = matrix_to_rot6d(torch.from_numpy(object_R_IMU.reshape(1, 3, 3)).to(device))
            except FileNotFoundError:
                object_init_T = keypoints3d[:, :3]
                object_init_R = matrix_to_rot6d(torch.from_numpy(object_R_IMU.reshape(1, 3, 3)).to(device))
                aaa = 1
        else:
            # object_init_T = object_T_reference
            # object_init_R = object_R_reference
            object_init_T = keypoints3d[:, :3]
            object_init_R = matrix_to_rot6d(torch.from_numpy(object_R_IMU.reshape(1, 3, 3)).to(device))

    faces = faces_idx.verts_idx
    verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
    textures = TexturesVertex(verts_features=verts_rgb.to(device))
    # Create a Meshes object for the teapot. Here we have only one mesh in the batch.
    chair_mesh = Meshes(
        verts=[verts.to(device)],
        faces=[faces.to(device)],
        textures=textures
    )
    chair_mesh = chair_mesh.extend(num_view)
    silhouette_renderer = nr.renderer.Renderer(
                image_size=256,
                K=K_subs,
                R=R_subs,
                t=T_subs,
                orig_size=256,
                anti_aliasing=False,
            )

    Rs = torch.from_numpy(R_subs.transpose(0,2,1))
    # Rs = torch.from_numpy(R_subs)
    Ts = torch.from_numpy(T_subs[:,0])
    # Initialize a model using the renderer, mesh and reference image
    model = ObjectPoseFittingBatch(meshes=chair_mesh, renderers=silhouette_renderer, image_refs=image_refs, Rs=Rs, Ts=Ts, Ks=K_subs, object_masks_valid=object_masks_valid, image_refs_background=image_refs_background, object_R=object_init_R, object_init_T=object_init_T).to(device)

    if object_T_reference is not None:
        num_iter = 600  #  600
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        begin_iter = 0.1
    else:
        num_iter = 2000  #  8000
        # num_iter = 600  # 8000
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        begin_iter = 0.1

    loss_pre = 0.
    loss = 0.
    loop = tqdm(total=num_iter)
    for i in range(num_iter):
        optimizer.zero_grad()
        loss_pre = loss
        loss, images = model()
        loss.backward()
        optimizer.step()
        if i == int(begin_iter * num_iter):
            if args.initial_search == True:
                rotations_initial = search_initial(model, image_refs)
                model = ObjectPoseFittingBatch(meshes=chair_mesh, renderers=silhouette_renderer, image_refs=image_refs,
                                               Rs=Rs, Ts=Ts, Ks=K_subs, object_masks_valid=object_masks_valid,
                                               image_refs_background=image_refs_background,
                                               object_R=object_init_R + rotations_initial,
                                               object_init_T=object_init_T).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
            model.rotations.requires_grad = True
        # if i >= int(0.5*num_iter) and abs(loss-loss_pre) <=0.1:
        #     break
        # print(loss.data)
        loop.set_description(f"loss: {loss.data.item():.6g}")
        loop.update()
    loop.close()

    object_T_reference = model.translations.detach().cpu().numpy()
    object_R_reference = (model.object_R_IMU + 1. * model.rotations).detach().cpu()

    object_T = model.translations.detach().cpu().numpy().tolist()
    object_R = (model.object_R_IMU + 1. * model.rotations).detach().cpu().numpy().tolist()


    object_RT = dict()
    object_RT['object_T'] = object_T
    object_RT['object_R'] = object_R

    if frame % seq_step == 0:
        with open(join(save_json_dir, '{}.json'.format(str(frame).zfill(6))), 'w') as json_file:
            json.dump(object_RT, json_file)

        mesh_temp = model.apply_transformation(model.rotations, model.translations)
        save_object_name = join(save_obj_dir, '{}.obj'.format(str(frame).zfill(6)))
        save_obj(save_object_name, verts=mesh_temp[0], faces=model.faces[0])

        print(frame)


    # for ind in range(num_view):
    #     image_grid([images[0][ind].cpu().detach().numpy(), image_refs[ind]], rows=1, cols=2 ,rgb=False)

