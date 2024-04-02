import matplotlib.pyplot as plt
import os.path as osp
import cv2
import numpy as np
from os.path import join


def load_mask(args, cams, frame):

    masks = []
    masks_center = []
    masks_background = []
    omask_dir_path = osp.join(args.root_path, args.foreground_path)

    for idx in cams:
        try:
            omask = cv2.imread(osp.join(omask_dir_path, '{}/'.format(str(idx)), '{}.jpg'.format(str(frame).zfill(6))), -1)/255.
        except TypeError:
            omask = np.zeros([1080,1920]).astype('float32')
        background = np.zeros_like(omask).astype('float32')
        for bp in args.background_path.split(','):
            background_dir_path = osp.join(args.root_path, bp)
            try:
                background += cv2.imread(osp.join(background_dir_path, '{}/'.format(idx), '{}.jpg'.format(str(frame).zfill(6))), -1)
            except TypeError:
                continue
        background[background>0]  = 1.
        masks_background.append(background)
        masks.append(omask.astype('float32'))
        omask = omask.astype('uint8')
        contours, hierarchy = cv2.findContours(omask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        # 找到最大区域并填充
        area = []
        for j in range(len(contours)):
            area.append(cv2.contourArea(contours[j]))
        # print(idx)
        if area != []:
            max_idx = np.argmax(area)
            max_area = cv2.contourArea(contours[max_idx])
            for k in range(len(contours)):
                if k != max_idx:
                    cv2.fillPoly(omask, [contours[k]], (0))
        masks_center.append(omask)

    return masks, omask, masks_center, masks_background

def load_mask_mpmo(args, root_path, cams, frame,object_idx):

    masks = []
    masks_center = []
    masks_background = []
    omask_dir_path = join(root_path, 'masks')

    for idx in cams:
        try:
            omask = cv2.imread(join(omask_dir_path, '{}/'.format(str(idx)), str(object_idx),'{}.png'.format(str(frame))), -1)/255.
        except TypeError:
            omask = np.zeros([1080,1920]).astype('float32')
        background = np.zeros_like(omask).astype('float32')
        for bp in args.background_path.split(','):
            background_dir_path = join(omask_dir_path, str(idx), bp)
            try:
                background += cv2.imread(join(background_dir_path, '{}.png'.format(str(frame))), -1)
            except TypeError:
                continue
        background[background>0]  = 1.
        masks_background.append(background)
        masks.append(omask.astype('float32'))
        omask = omask.astype('uint8')
        contours, hierarchy = cv2.findContours(omask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        # 找到最大区域并填充
        area = []
        for j in range(len(contours)):
            area.append(cv2.contourArea(contours[j]))
        # print(idx)
        if area != []:
            max_idx = np.argmax(area)
            max_area = cv2.contourArea(contours[max_idx])
            for k in range(len(contours)):
                if k != max_idx:
                    cv2.fillPoly(omask, [contours[k]], (0))
        masks_center.append(omask)

    return masks, omask, masks_center, masks_background



def mask_cam_zoom_in(num_view, camera_param, masks, masks_background, masks_center):

    K_subs = []
    maskNs = []
    maskNs_background = []
    P = []
    object_keypoints = []
    object_masks_valid = np.ones(num_view)
    R_subs = []
    T_subs = []
    for idx in range(0, num_view):
        try:
            K = np.array(camera_param[str(idx)]['K']).reshape((3, 3))
        except KeyError:
            K = np.array(camera_param['0']['K']).reshape((3, 3))
            object_masks_valid[idx] = 0
        K *= 0.5
        K[2, 2] = 1
        try:
            RT = camera_param[str(idx)]['RT'].copy()
        except KeyError:
            RT = camera_param['0']['RT'].copy()
        if len(RT) == 16:
            E = np.array(RT).reshape((4, 4))
        else:
            E = np.array(RT).reshape((3, 4))
        mask = masks[idx]
        mask_background = masks_background[idx]
        mask_center = masks_center[idx]
        Y, X = np.where(mask > 0)

        ##### height now #####

        loadsize = 256
        try:
            ori_height = Y.max() - Y.min()
            ori_width = X.max() - X.min()
        except ValueError:
            Y, X = np.where(mask == 0)
            ori_height = Y.max() - Y.min()
            ori_width = X.max() - X.min()

        H, W = mask.shape[:2]

        ## the mask is too small
        if ori_height / H < 0.01  or ori_width / W < 0.01:
            Y, X = np.where(mask >= 0)
            ori_height = Y.max() - Y.min()
            ori_width = X.max() - X.min()
            object_masks_valid[idx] = 0

        if ori_height > ori_width:
            set_height = 450
            scaleH = (set_height / ori_height)
            scaleF = loadsize * scaleH / 512
            Ori_shift = abs(H - W) // 2
            Add_shift = int(500)

            if H > W:
                maskP = np.pad(mask, ((Add_shift, Add_shift), ((H - W) // 2 + Add_shift, (H - W) // 2 + Add_shift)),
                               'constant', constant_values=(0, 0))
                maskP_background = np.pad(mask_background,
                                          (
                                              (Add_shift, Add_shift),
                                              ((H - W) // 2 + Add_shift, (H - W) // 2 + Add_shift)),
                                          'constant', constant_values=(0, 0))
                maskP_center = np.pad(mask_center,
                                      ((Add_shift, Add_shift), ((H - W) // 2 + Add_shift, (H - W) // 2 + Add_shift)),
                                      'constant', constant_values=(0, 0))
            elif H < W:
                maskP = np.pad(mask, (((W - H) // 2 + Add_shift, (W - H) // 2 + Add_shift), (Add_shift, Add_shift)),
                               'constant', constant_values=(0, 0))
                maskP_background = np.pad(mask_background,
                                          (
                                              ((W - H) // 2 + Add_shift, (W - H) // 2 + Add_shift),
                                              (Add_shift, Add_shift)),
                                          'constant', constant_values=(0, 0))
                maskP_center = np.pad(mask_center,
                                      (((W - H) // 2 + Add_shift, (W - H) // 2 + Add_shift), (Add_shift, Add_shift)),
                                      'constant', constant_values=(0, 0))
            else:
                maskP = mask
                maskP_background = mask_background
                maskP_center = mask_center

            E_2 = int(512 / scaleH) // 2
            My = (np.max(Y) - np.min(Y)) // 2 + np.min(Y)
            Mx = (np.max(X) - np.min(X)) // 2 + np.min(X)

            if H > W:
                maskN = maskP[Add_shift + My - E_2: Add_shift + My + E_2,
                        Add_shift + Ori_shift + Mx - E_2:  Add_shift + Ori_shift + Mx + E_2]
                maskN_background = maskP_background[Add_shift + My - E_2: Add_shift + My + E_2,
                                   Add_shift + Ori_shift + Mx - E_2:  Add_shift + Ori_shift + Mx + E_2]
                maskN_center = maskP_center[Add_shift + My - E_2: Add_shift + My + E_2,
                               Add_shift + Ori_shift + Mx - E_2:  Add_shift + Ori_shift + Mx + E_2]
                left_top_x = Mx - E_2
                left_top_y = My - E_2
            elif H < W:
                maskN = maskP[Add_shift + Ori_shift + My - E_2: Add_shift + Ori_shift + My + E_2,
                        Add_shift + Mx - E_2:  Add_shift + Mx + E_2]
                maskN_background = maskP_background[Add_shift + Ori_shift + My - E_2: Add_shift + Ori_shift + My + E_2,
                                   Add_shift + Mx - E_2:  Add_shift + Mx + E_2]
                maskN_center = maskP_center[Add_shift + Ori_shift + My - E_2: Add_shift + Ori_shift + My + E_2,
                               Add_shift + Mx - E_2:  Add_shift + Mx + E_2]
                left_top_x = Mx - E_2
                left_top_y = My - E_2
            else:
                maskN = maskP
                maskN_background = maskP_background
                maskN_center = maskP_center

                left_top_x = 0
                left_top_y = 0
        else:
            set_width = 450

            scaleF = (set_width / ori_width)
            scaleH = loadsize * scaleF / 512


            Ori_shift = abs(H - W) // 2
            Add_shift = int(150)

            if H > W:
                maskP = np.pad(mask, ((Add_shift, Add_shift), ((H - W) // 2 + Add_shift, (H - W) // 2 + Add_shift)),
                               'constant', constant_values=(0, 0))
                maskP_background = np.pad(mask_background,
                                          (
                                              (Add_shift, Add_shift),
                                              ((H - W) // 2 + Add_shift, (H - W) // 2 + Add_shift)),
                                          'constant', constant_values=(0, 0))
                maskP_center = np.pad(mask_center,
                                      ((Add_shift, Add_shift), ((H - W) // 2 + Add_shift, (H - W) // 2 + Add_shift)),
                                      'constant', constant_values=(0, 0))
            elif H < W:
                maskP = np.pad(mask, (((W - H) // 2 + Add_shift, (W - H) // 2 + Add_shift), (Add_shift, Add_shift)),
                               'constant', constant_values=(0, 0))
                maskP_background = np.pad(mask_background,
                                          (
                                              ((W - H) // 2 + Add_shift, (W - H) // 2 + Add_shift),
                                              (Add_shift, Add_shift)),
                                          'constant', constant_values=(0, 0))
                maskP_center = np.pad(mask_center,
                                      (((W - H) // 2 + Add_shift, (W - H) // 2 + Add_shift), (Add_shift, Add_shift)),
                                      'constant', constant_values=(0, 0))
            else:
                maskP = mask
                maskP_background = mask_background
                maskP_center = mask_center

            # E_2 = int(512 / scaleH) // 2
            E_2 = int(512 / scaleF) // 2

            My = (np.max(Y) - np.min(Y)) // 2 + np.min(Y)
            Mx = (np.max(X) - np.min(X)) // 2 + np.min(X)

            if H > W:
                maskN = maskP[Add_shift + My - E_2: Add_shift + My + E_2,
                        Add_shift + Ori_shift + Mx - E_2:  Add_shift + Ori_shift + Mx + E_2]
                maskN_background = maskP_background[Add_shift + My - E_2: Add_shift + My + E_2,
                                   Add_shift + Ori_shift + Mx - E_2:  Add_shift + Ori_shift + Mx + E_2]
                maskN_center = maskP_center[Add_shift + My - E_2: Add_shift + My + E_2,
                               Add_shift + Ori_shift + Mx - E_2:  Add_shift + Ori_shift + Mx + E_2]
                left_top_x = Mx - E_2
                left_top_y = My - E_2
            elif H < W:
                maskN = maskP[Add_shift + Ori_shift + My - E_2: Add_shift + Ori_shift + My + E_2,
                        Add_shift + Mx - E_2:  Add_shift + Mx + E_2]
                maskN_background = maskP_background[Add_shift + Ori_shift + My - E_2: Add_shift + Ori_shift + My + E_2,
                                   Add_shift + Mx - E_2:  Add_shift + Mx + E_2]
                maskN_center = maskP_center[Add_shift + Ori_shift + My - E_2: Add_shift + Ori_shift + My + E_2,
                               Add_shift + Mx - E_2:  Add_shift + Mx + E_2]
                left_top_x = Mx - E_2
                left_top_y = My - E_2
            else:
                maskN = maskP
                maskN_background = maskP_background
                maskN_center = maskP_center

                left_top_x = 0
                left_top_y = 0


        cx_o = K[0, 2]
        cy_o = K[1, 2]
        # print(cx_o, cy_o)
        cx_n = cx_o - left_top_x
        cy_n = cy_o - left_top_y
        # print(cx_n, cy_n)
        K_n = K.copy()
        K_n[0, 2] = cx_n
        K_n[1, 2] = cy_n

        K_n_s = K_n.copy()
        if ori_height > ori_width:
            K_n_s *= scaleF
        else:
            K_n_s *= scaleH

        K_n_s[2, 2] = 1
        maskN = cv2.resize(maskN, tuple((loadsize, loadsize)), interpolation=cv2.INTER_NEAREST)
        maskN_background = cv2.resize(maskN_background, tuple((loadsize, loadsize)),
                                      interpolation=cv2.INTER_NEAREST)
        maskN_center = cv2.resize(maskN_center, tuple((loadsize, loadsize)), interpolation=cv2.INTER_NEAREST)

        cnts = cv2.findContours(maskN_center, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0]
        if maskN_center.sum() == 0:
            center_point = np.array((0, 0, 0)).reshape(1, 3)
        else:
            # 使用外切矩形法求得几何中点
            x, y, w, h = cv2.boundingRect(cnts[0])
            cX = x + w // 2
            cY = y + h // 2
            center_point = np.array((cX, cY, 1)).reshape(1, 3)

        object_keypoints.append(center_point)
        maskNs.append(maskN)
        maskNs_background.append(maskN_background)
        K_subs.append(K_n_s)
        P.append(K_n_s @ E[:3,:])
        if len(RT) == 16:
            R_subs.append(np.array(RT).reshape(4, 4)[:3, :3])
            T_subs.append(np.array(RT).reshape(4, 4)[:3, 3:].reshape(1, 3))
        else:
            R_subs.append(np.array(RT).reshape(3, 4)[:3, :3])
            T_subs.append(np.array(RT).reshape(3, 4)[:3, 3:].reshape(1, 3))

    K_subs = np.stack(K_subs)
    R_subs = np.stack(R_subs)
    T_subs = np.stack(T_subs)
    return maskNs, maskNs_background, K_subs, R_subs, T_subs, P, object_masks_valid, object_keypoints



