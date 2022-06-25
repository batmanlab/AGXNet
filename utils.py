"""Utility functions."""
import numpy as np
import imageio
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from scipy import ndimage
import json
import cv2
from imgaug import augmenters as iaa


class BoundingBoxGenerator:
    """Generate bounding box. """

    def __init__(self, heatmap, mode='hot', percentile=0.95, min_obj_size=1):
        self.heatmap = heatmap
        self.mode = mode
        self.percentile = percentile
        self.min_obj_size = min_obj_size  # number of pixels in the object

    def get_bbox_pct(self):
        # create quantile mask
        if self.mode == 'hot':
            q = np.quantile(self.heatmap, self.percentile)
            mask = self.heatmap > q
        elif self.mode == 'cold':
            q = np.quantile(self.heatmap, 1 - self.percentile)
            mask = self.heatmap < q
        else:
            raise Exception('Invalid mode.')

        # label connected pixels in the mask
        label_im, nb_labels = ndimage.label(mask)

        # find the sizes of connected pixels
        sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))

        # create labeled image
        mask_size = sizes < self.min_obj_size
        remove_pixel = mask_size[label_im]
        label_im[remove_pixel] = 0
        labels = np.unique(label_im)
        label_im = np.searchsorted(labels, label_im)  # sort objects from large to small

        # generate bounding boxes
        bbox = []
        for l in range(1, len(labels)):
            slice_x, slice_y = ndimage.find_objects(label_im == l)[0]
            if (slice_x.start < slice_x.stop) & (slice_y.start < slice_y.stop):
                b = [slice_y.start, slice_x.start, slice_y.stop, slice_x.stop]
                bbox.append(b)
        return bbox


def get_aug_bbox(image_path, bb_tuple, resize):
    """
    get the coordinates of bounding box on the augmented image.
    :param img_path: path of the original image
    :param bb_org: tuple of original bbox inputs and type, which is either 'xywh' or 'x1y1x2y2'
    :param augmentations:
    :return: bounding box [x1, y1, x2, y2] after transformation / augmentation
    """
    image_org = imageio.imread(image_path)
    h, w = image_org.shape
    if h >= w:
        affine_trans = iaa.Sequential([iaa.Resize({'width': resize, 'height': 'keep-aspect-ratio'}),
                                       iaa.CenterCropToFixedSize(width=resize, height=resize)])
    else:
        affine_trans = iaa.Sequential(
            [iaa.Resize({'height': resize, 'width': 'keep-aspect-ratio'}),
             iaa.CenterCropToFixedSize(width=resize, height=resize)])

    bb = bb_tuple[0]
    if bb_tuple[1] == 'xywh':
        bb_ia = BoundingBoxesOnImage([
            BoundingBox(x1=bb[0], x2=bb[0]+bb[2], y1=bb[1], y2=bb[1]+bb[3]),
        ], shape=image_org.shape)
    elif bb_tuple[1] == 'x1y1x2y2':
        bb_ia = BoundingBoxesOnImage([
            BoundingBox(x1=bb[0], x2=bb[2], y1=bb[1], y2=bb[3]),
        ], shape=image_org.shape)
    else:
        raise Exception('Bounding box type is not allowed.')
    image_aug, bb_aug = affine_trans(image=image_org, bounding_boxes=bb_ia)
    return [bb_aug[0].x1, bb_aug[0].y1, bb_aug[0].x2, bb_aug[0].y2]


def get_landmark_idx(args):
    """get the landmark index."""

    with open(args.landmark_dictionary) as f:
        dict_landmark = json.load(f)

    with open(args.landmark_mapping) as f:
        dict_mapping = json.load(f)

    landmark_lst = [v for k, v in dict_landmark.items()]

    idx_lst = []
    for k,v in dict_mapping.items():
        idx_lst.append(landmark_lst.index(v))

    return idx_lst


def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : 1D array (x1, y1, x2, y2)
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : 1D array (x1, y1, x2, y2)
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    Returns
    -------
        iou, iobb
    """
    assert bb1[0] < bb1[2]
    assert bb1[1] < bb1[3]
    assert bb2[0] < bb2[2]
    assert bb2[1] < bb2[3]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    if x_right < x_left or y_bottom < y_top:
        intersection_area = 0
    else:
        intersection_area = float((x_right - x_left) * (y_bottom - y_top))

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    union_area = float(bb1_area + bb2_area - intersection_area)
    iou = intersection_area / union_area
    assert iou >= 0.0
    assert iou <= 1.0
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction area
    iobb = intersection_area / float(bb2_area)
    assert iobb >= 0.0
    assert iobb <= 1.0
    return iou, iobb


def get_iobb(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    Calculate the Intersection of the GT over the detected bounding box IoBB1.
    Calculate the Intersection of the detected bounding box over the GT IoBB2.

    Parameters
    ----------
    bb1 : 1D array (x1, y1, x2, y2)
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : 1D array (x1, y1, x2, y2)
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    Returns
    -------
        iou, iobb
    """
    assert bb1[0] < bb1[2]
    assert bb1[1] < bb1[3]
    assert bb2[0] < bb2[2]
    assert bb2[1] < bb2[3]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    if x_right < x_left or y_bottom < y_top:
        intersection_area = 0
    else:
        intersection_area = float((x_right - x_left) * (y_bottom - y_top))

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    union_area = float(bb1_area + bb2_area - intersection_area)
    iou = intersection_area / union_area
    assert iou >= 0.0
    assert iou <= 1.0
    # compute the intersection over the ground truth bounding box, i.e., bb1
    iobb1 = intersection_area / float(bb1_area)
    assert iobb1 >= 0.0
    assert iobb1 <= 1.0
    # compute the intersection over the detected bounding box, i.e., bb2
    iobb2 = intersection_area / float(bb2_area)
    assert iobb2 >= 0.0
    assert iobb2 <= 1.0
    return iou, iobb1, iobb2


def flatten(t):
    return [item for sublist in t for item in sublist]


def im2double(im):
    return cv2.normalize(im.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)


def get_cumlative_attention(cam, bbox):
    return cam[bbox[1]:bbox[3], bbox[0]:bbox[2]].sum()


def get_largest_k_indices_in_list(lst, k):
    return sorted(range(len(lst)), key=lambda i: lst[i], reverse=True)[:2]


def get_smallest_k_indices_in_list(lst, k):
    return sorted(range(len(lst)), key=lambda i: lst[i])[-2:]


def get_topk_bbox(bbox, cam, k):
    """
    get bounding boxes with the top k largest cumulative attention scores
    :param bbox:
    :param cam:
    :param k:
    :return:
    """
    bb_cum_att = [get_cumlative_attention(cam, b) for b in bbox]
    sorted_indices = get_largest_k_indices_in_list(bb_cum_att, k)
    bb_sorted = [bbox[j] for j in sorted_indices[:k]]
    return bb_sorted


def label_tp(x, threshold):
    if x > threshold:
        return 1
    else:
        return 0


def evaluate_detected_bbox(df_sel, metric, threshold):
    if metric == 'IoBB':
        df_sel['tp'] = df_sel.apply(lambda x: label_tp(x.IoBB, threshold), axis=1)
    elif metric == 'IoU':
        df_sel['tp'] = df_sel.apply(lambda x: label_tp(x.IoU, threshold), axis=1)
    else:
        raise Exception('Invalid objection detection metric.')

    # ground truth
    df_gt_cnt = df_sel.groupby(['LANDMARK_IMAGENOME'])['DICOM_ID'].nunique().reset_index()
    df_gt_cnt.columns = ['LANDMARK_IMAGENOME', 'GT_CNT']

    # detected box
    df_det_cnt = df_sel.groupby(['LANDMARK_IMAGENOME']).size().reset_index()
    df_det_cnt.columns = ['LANDMARK_IMAGENOME', 'DETECTED_CNT']

    # True Positive
    df_tp_cnt = df_sel.groupby(['LANDMARK_IMAGENOME']).sum()['tp'].reset_index()
    df_tp_cnt.columns = ['LANDMARK_IMAGENOME', 'TP_CNT']

    # Recall
    df_metric = df_gt_cnt.merge(df_det_cnt, on='LANDMARK_IMAGENOME')
    df_metric = df_metric.merge(df_tp_cnt, on='LANDMARK_IMAGENOME')
    df_metric['RECALL'] = df_metric['TP_CNT'] / df_metric['GT_CNT']
    df_metric['PRECISION'] = df_metric['TP_CNT'] / df_metric['DETECTED_CNT']

    avg_recall = df_metric.sum()['TP_CNT'] / df_metric.sum()['GT_CNT']
    avg_precision = df_metric.sum()['TP_CNT'] / df_metric.sum()['DETECTED_CNT']

    return df_metric, avg_recall, avg_precision


def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.
    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


def iou(row):
    iou, iobb1, iobb2 = get_iobb(row['bb_gt'], row['bb_gen'])
    return iou


def iobb1(row):
    iou, iobb1, iobb2 = get_iobb(row['bb_gt'], row['bb_gen'])
    return iobb1


def iobb2(row):
    iou, iobb1, iobb2 = get_iobb(row['bb_gt'], row['bb_gen'])
    return iobb2


def bb_gen_area(row):
    a = (row['bb_gen'][2] - row['bb_gen'][0]) * (row['bb_gen'][3] - row['bb_gen'][1])
    return a


def normalize_cam(cam1):
    cam1 -= cam1.min()
    cam1 /= (cam1.max() + 1e-12) # pervent from dividing 0
    return cam1


def get_recall_precision(df, iou_thres):
    """Compute recall and precision."""

    df_gen = df.groupby(['dicom_id', 'bb_gen_idx'])['iou'].max().reset_index()
    df_gt = df.groupby(['dicom_id', 'bb_gt_idx'])['iou'].max().reset_index()
    recall_lst = []
    prec_lst = []
    for iou in iou_thres:
        idx_tp = df_gen['iou'] >= iou
        tp = idx_tp.sum()
        idx_fp = df_gen['iou'] < iou
        fp = idx_fp.sum()
        idx_fn = df_gt['iou'] < iou
        fn = idx_fn.sum()

        recall = tp / (tp + fn)
        recall_lst.append(recall)

        prec = tp / (tp + fp)
        prec_lst.append(prec)

    return np.array(recall_lst), np.array(prec_lst)


def average_precision(recalls, precisions, mode='area'):
    """Calculate average precision (for single or multiple scales).
    Args:
        recalls (ndarray): shape (num_scales, num_dets) or (num_dets, )
        precisions (ndarray): shape (num_scales, num_dets) or (num_dets, )
        mode (str): 'area' or '11points', 'area' means calculating the area
            under precision-recall curve, '11points' means calculating
            the average precision of recalls at [0, 0.1, ..., 1]
    Returns:
        float or ndarray: calculated average precision
    """
    no_scale = False
    if recalls.ndim == 1:
        no_scale = True
        recalls = recalls[np.newaxis, :]
        precisions = precisions[np.newaxis, :]
    assert recalls.shape == precisions.shape and recalls.ndim == 2
    num_scales = recalls.shape[0]
    ap = np.zeros(num_scales, dtype=np.float32)
    if mode == 'area':
        zeros = np.zeros((num_scales, 1), dtype=recalls.dtype)
        ones = np.ones((num_scales, 1), dtype=recalls.dtype)
        mrec = np.hstack((zeros, recalls, ones))
        mpre = np.hstack((zeros, precisions, zeros))
        for i in range(mpre.shape[1] - 1, 0, -1):
            mpre[:, i - 1] = np.maximum(mpre[:, i - 1], mpre[:, i])
        for i in range(num_scales):
            ind = np.where(mrec[i, 1:] != mrec[i, :-1])[0]
            ap[i] = np.sum(
                (mrec[i, ind + 1] - mrec[i, ind]) * mpre[i, ind + 1])
    elif mode == '11points':
        for i in range(num_scales):
            for thr in np.arange(0, 1 + 1e-3, 0.1):
                precs = precisions[i, recalls[i, :] >= thr]
                prec = precs.max() if precs.size > 0 else 0
                ap[i] += prec
        ap /= 11
    else:
        raise ValueError(
            'Unrecognized mode, only "area" and "11points" are supported')
    if no_scale:
        ap = ap[0]
    return ap


def keep_dicom_idx_in_batch(input_dicom_lst, target_lst):
    idx_lst = []
    for i in range(len(input_dicom_lst)):
        did = input_dicom_lst[i]
        # works for discard
        if not (did in target_lst):
            idx_lst.append(i)
    return idx_lst