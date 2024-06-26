import cv2
import numpy as np
import PIL.Image as pil_img
import matplotlib.pyplot as plt


def get_one_box(det_output, thrd=0.9):
    max_area = 0
    max_bbox = None
    for i in range(det_output['boxes'].shape[0]):
        bbox = det_output['boxes'][i]
        score = det_output['scores'][i]
        if float(score) < thrd:
            continue
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if float(area) > max_area:
            max_bbox = [float(x) for x in bbox]
            max_area = area

    return max_bbox

def get_topk_box(det_output, thrd=0.9, k=4):
    # max_area = 0
    # max_bbox = None
    # topk_bbox = None
    area_list = []
    bbox_list = []
    for i in range(det_output['boxes'].shape[0]):
        bbox = det_output['boxes'][i]
        score = det_output['scores'][i]
        if float(score) < thrd:
            continue
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        area_list.append(area.detach().cpu().numpy())
        bbox_list.append([float(x) for x in bbox])
        # if float(area) > max_area:
        #     max_bbox = [float(x) for x in bbox]
        #     max_area = area

    area_list = np.array(area_list)
    index = area_list.argsort()[-k:][::-1]  # 获取前k个索引
    topk_bbox = [bbox_list[x] for x in index]
    return topk_bbox


def get_max_iou_box(det_output, prev_bbox, thrd=0.9):
    max_score = 0
    max_bbox = None
    for i in range(det_output['boxes'].shape[0]):
        bbox = det_output['boxes'][i]
        score = det_output['scores'][i]
        # if float(score) < thrd:
        #     continue
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        iou = calc_iou(prev_bbox, bbox)
        iou_score = float(score) * iou
        if float(iou_score) > max_score:
            max_bbox = [float(x) for x in bbox]
            max_score = iou_score

    return max_bbox


def calc_iou(bbox1, bbox2):
    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]

    xA = max(bbox1[0], bbox2[0])
    yA = max(bbox1[1], bbox2[1])
    xB = min(bbox1[2], bbox2[2])
    yB = min(bbox1[3], bbox2[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    box1Area = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
    box2Area = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)
    
    iou = interArea / float(box1Area + box2Area - interArea)

    return iou


def vis_bbox(img_path, bbox):

    x1, y1, x2, y2 = bbox

    image = cv2.imread(img_path)
    draw1 = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    cv2.imshow('draw 1', draw1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def vis_smpl_3d(pose_output, img, cam_root, f, c, renderer, color_id=0, cam_rt=np.zeros(3),
                cam_t=np.zeros(3), J_regressor_h36m=None):
    '''
    input theta_mats: np.ndarray (96, )
    input betas: np.ndarray (10, )
    input img: RGB Image array with value in [0, 1]
    input cam_root: np.ndarray (3, )
    input f: np.ndarray (2, )
    input c: np.ndarray (2, )
    '''

    # vertices = pose_output.pred_vertices.detach().cpu().numpy().squeeze()
    # vertices = pose_output.detach().cpu().numpy().squeeze()
    # vertices = pose_output.pred_vertices.detach().cpu().numpy().squeeze()
    vertices = pose_output
    # J_from_verts_h36m = vertices2joints(J_regressor_h36m, pose_output['vertices'].detach().cpu())

    # cam_for_render = np.hstack([f[0], c])

    # center = pose_output.joints[0][0].cpu().data.numpy()
    # vert_shifted = vertices - center + cam_root
    # vert_shifted = vertices + cam_root
    vert_shifted = vertices
    vert_shifted = vert_shifted

    # Render results
    rend_img_overlay = renderer(
        vert_shifted, princpt=c, img=img, do_alpha=True, color_id=color_id, cam_rt=cam_rt, cam_t=cam_t)

    img = pil_img.fromarray(rend_img_overlay[:, :, :3].astype(np.uint8))
    # if len(filename) > 0:
    #     img.save(filename)

    return np.asarray(img)


def show(image):
    plt.imshow(image)
    plt.show()
