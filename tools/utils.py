import tensorflow as tf
import numpy as np
import os
import skimage
import cv2
from math import cos, sin
from imgaug import augmenters as iaa
import imgaug as ia
from tensorflow.contrib import slim


def restore_ckpt(sess: tf.Session, var_list: list, ckptdir: str):
    if ckptdir == '' or ckptdir == None:
        pass
    elif 'mobilenet' in ckptdir:
        variables_to_restore = slim.get_model_variables()
        loader = tf.train.Saver([var for var in variables_to_restore if 'MobilenetV1' in var.name])
        loader.restore(sess, ckptdir)
    else:
        ckpt = tf.train.get_checkpoint_state(ckptdir)
        loader = tf.train.Saver(var_list=var_list)
        loader.restore(sess, ckpt.model_checkpoint_path)


def write_arguments_to_file(args, filename):
    with open(filename, 'w') as f:
        for key, value in vars(args).items():
            f.write('%s: %s\n' % (key, str(value)))


class Helper(object):
    def __init__(self, image_list: str, box_list: str, class_num: int, anchors: str, in_hw: tuple, out_hw: tuple):
        self.in_h = in_hw[0]
        self.in_w = in_hw[1]
        self.out_h = out_hw[0]
        self.out_w = out_hw[1]
        if box_list:
            self.box_list = np.loadtxt(box_list, dtype=str)
            self.total_data = len(self.box_list)
        else:
            self.box_list = None
        if box_list:
            self.image_list = np.loadtxt(image_list, dtype=str)
        else:
            self.image_list = None
        self.grid_w = 1/self.out_w
        self.grid_h = 1/self.out_h
        if class_num:
            self.class_num = class_num
        if anchors:
            self.anchors = np.loadtxt(anchors, ndmin=2)
            self.wh_scale = self._anchor_scale()
            self.xy_offset = self._coordinate_offset()
        self.iaaseq = iaa.Sequential([
            iaa.Fliplr(0.5),  # 50% 镜像
            iaa.Add((-30, 30), per_channel=0.5),
            iaa.Crop(percent=(0, 0.1)),  # random crops
            # Strengthen or weaken the contrast in each image.
            iaa.ContrastNormalization((0.5, 1.5)),
            # which can end up changing the color of the images.
            iaa.Multiply((0.6, 1.4), per_channel=0.2),
            iaa.GaussianBlur(sigma=(0.0, 2.0)),
            # Apply affine transformations to each image.
            # Scale/zoom them, translate/move them, rotate them and shear them.
            iaa.Affine(scale={"x": (0.8, 1.4), "y": (0.8, 1.4)},
                       translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                       rotate=(-10, 10))
        ])
        self.colormap = [(128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                         (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                         (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                         (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128),
                         (61, 230, 250), (255, 6, 51), (11, 102, 255), (255, 7, 71), (255, 9, 224),
                         (9, 7, 230), (220, 220, 220), (255, 9, 92), (112, 9, 255), (8, 255, 214),
                         (7, 255, 224), (255, 184, 6), (10, 255, 71), (255, 41, 10), (7, 255, 255),
                         (224, 255, 8), (102, 8, 255), (255, 61, 6), (255, 194, 7), (255, 122, 8),
                         (0, 255, 20), (255, 8, 41), (255, 5, 153), (6, 51, 255), (235, 12, 255),
                         (160, 150, 20), (0, 163, 255), (140, 140, 140), (250, 10, 15), (20, 255, 0),
                         (31, 255, 0), (255, 31, 0), (255, 224, 0), (153, 255, 0), (0, 0, 255),
                         (255, 71, 0), (0, 235, 255), (0, 173, 255), (31, 0, 255), (11, 200, 200),
                         (255, 82, 0), (0, 255, 245), (0, 61, 255), (0, 255, 112), (0, 255, 133),
                         (255, 0, 0), (255, 163, 0), (255, 102, 0), (194, 255, 0), (0, 143, 255),
                         (51, 255, 0), (0, 82, 255), (0, 255, 41), (0, 255, 173), (10, 0, 255),
                         (173, 255, 0), (0, 255, 153), (255, 92, 0), (255, 0, 255), (255, 0, 245)]

    def _xy_to_grid(self, box: np.ndarray)->tuple:
        if box[1] == 1.0:
            idx, modx = self.out_w-1, 1.0
        else:
            idx, modx = divmod(box[1], self.grid_w)
            modx /= self.grid_w

        if box[2] == 1.0:
            idy, mody = self.out_h-1, 1.0
        else:
            idy, mody = divmod(box[2], self.grid_h)
            mody /= self.grid_h
        return int(idx), modx, int(idy), mody

    def _wh_to_grid(self, box: np.ndarray)->tuple:
        w = box[3]/self.grid_w
        h = box[4]/self.grid_h
        return w, h

    def _fake_iou(self, wh, wh1):
        s1 = wh[0]*wh[1]
        s2 = wh1[0]*wh1[1]
        iner = np.minimum(wh[0], wh1[0])*np.minimum(wh[1], wh1[1])
        return iner/(s1+s2-iner)

    def _get_anchor_index(self, wh):
        iouvalue = np.zeros(len(self.anchors))
        for i, anchor in enumerate(self.anchors):
            iouvalue[i] = self._fake_iou(wh, anchor)
        return np.argmax(iouvalue)

    def box_to_label(self, true_box):
        label = np.zeros((self.out_h, self.out_w, len(self.anchors), 5+self.class_num))
        for box in true_box:
            # remove small box
            # if box[2] <= .1 or box[3] <= .1:
            #     continue
            idx, modx, idy, mody = self._xy_to_grid(box)
            w, h = self._wh_to_grid(box)
            anc_idx = self._get_anchor_index((w, h))
            label[idy, idx, anc_idx, 0] = modx  # x
            label[idy, idx, anc_idx, 1] = mody  # y
            label[idy, idx, anc_idx, 2] = w/self.anchors[anc_idx, 0]  # w
            label[idy, idx, anc_idx, 3] = h/self.anchors[anc_idx, 1]  # h
            label[idy, idx, anc_idx, 4] = 1.
            label[idy, idx, anc_idx, 5+int(box[0])] = 1.
        return label

    def _coordinate_offset(self):
        offset = np.zeros((self.out_h, self.out_w, len(self.anchors), 2))
        for i in range(self.out_h):
            for j in range(self.out_w):
                offset[i, j, :] = np.array([j, i])  # NOTE  [x,y]
        offset[..., 0] /= self.out_w
        offset[..., 1] /= self.out_h
        return offset

    def _anchor_scale(self):
        scale = np.zeros((self.out_h, self.out_w, len(self.anchors), 2))
        for i in range(len(self.anchors)):
            scale[:, :, i, :] = np.array(self.anchors[i])*np.array([self.grid_w, self.grid_h])
        return scale

    def _xy_to_all(self, label: np.ndarray):
        label[..., 0:2] = label[..., 0:2] * np.array([self.grid_w, self.grid_h])+self.xy_offset

    def _wh_to_all(self, label: np.ndarray):
        label[..., 2:4] = label[..., 2: 4]*self.wh_scale

    def label_to_box(self, label):
        self._xy_to_all(label)
        self._wh_to_all(label)
        true_box = label[np.where(label[..., 4] > .7)]
        p = np.argmax(true_box[:, 5:], axis=-1)
        true_box = np.c_[p, true_box[:, :4]]
        return true_box

    def data_augmenter(self, img: np.ndarray, true_box: np.ndarray)->tuple:
        """ augmenter for image 

        Parameters
        ----------
        img : np.ndarray
            img src
        true_box : np.ndarray
            box

        Returns
        -------
        tuple
            [image src,box] after data augmenter
        """
        seq_det = self.iaaseq.to_deterministic()
        img = img.astype('uint8')
        p = true_box[:, 0:1]
        xywh_box = true_box[:, 1:]

        bbs = ia.BoundingBoxesOnImage.from_xyxy_array(self.center_to_corner(xywh_box), shape=(self.in_h, self.in_w))

        image_aug = seq_det.augment_images([img])[0]
        bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
        bbs_aug = bbs_aug.remove_out_of_image().clip_out_of_image()

        xyxy_box = bbs_aug.to_xyxy_array()
        new_box = self.corner_to_center(xyxy_box)
        new_box = np.hstack((p[0:new_box.shape[0], :], new_box))
        return image_aug, new_box

    def _read_box(self, ann_path: str)->np.ndarray:
        return np.loadtxt(ann_path, dtype=float, ndmin=2)

    def _read_img(self, img_path: str, is_resize: bool)->np.ndarray:
        img = skimage.io.imread(img_path)
        if len(img.shape) != 3:
            img = skimage.color.gray2rgb(img)
        if is_resize:
            img = skimage.transform.resize(img, (self.in_h, self.in_w), mode='reflect', preserve_range=True)
        return img

    def _process_img(self, img: np.ndarray, true_box: np.ndarray, is_training: bool)->tuple:
        """ process image and true box , if is training then use data augmenter

        Parameters
        ----------
        img : np.ndarray
            image srs
        true_box : np.ndarray
            box 
        is_training : bool
            wether to use data augmenter

        Returns
        -------
        tuple
            image src , true box
        """
        if is_training:
            img, true_box = self.data_augmenter(img, true_box)

        # normlize image
        img = img/np.max(img)
        return img, true_box

    def generator(self, is_training=True, is_resize=True, is_make_lable=True):
        for i in range(len(self.box_list)):
            true_box = self._read_box(self.box_list[i])
            img = self._read_img(self.image_list[i], is_resize)
            img, true_box = self._process_img(img, true_box, is_training)
            if is_make_lable:
                yield img, self.box_to_label(true_box)
            else:
                yield img, true_box

    def _dataset_parser(self, img_path, box_path, is_training: bool, is_resize: bool):
        img = self._read_img(img_path.decode(), is_resize)
        true_box = self._read_box(box_path.decode())
        img, true_box = self._process_img(img, true_box, is_training)
        label = self.box_to_label(true_box)
        return img.astype('float32'), label.astype('float32')

    def set_dataset(self, batch_size, rand_seed, is_training=True, is_resize=True):
        def parser(img_path, box_path): return self._dataset_parser(img_path, box_path, is_training, is_resize)

        # dataset = tf.data.Dataset.from_generator(self._dataset_generator, (tf.string, tf.float32),
        #                                          (tf.TensorShape([]), tf.TensorShape([None, 4])))

        dataset = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(self.image_list), tf.convert_to_tensor(self.box_list)))

        dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(batch_size*10, count=None, seed=rand_seed))

        dataset = dataset.apply(tf.data.experimental.map_and_batch(
            map_func=lambda img_path, box_path: tuple(tf.py_func(parser, [img_path, box_path], [tf.float32, tf.float32])),
            batch_size=batch_size, drop_remainder=True))

        dataset = dataset.prefetch(3)

        self.dataset = dataset
        self.batch_size = batch_size
        self.epoch_step = self.total_data//batch_size

    def get_iter(self):
        return self.dataset.make_one_shot_iterator().get_next()

    def draw_box(self, img, true_box):
        """ [x,y,w,h,1] """
        p = true_box[:, 0]
        xyxybox = self.center_to_corner(true_box[:, 1:])
        for i, a in enumerate(xyxybox):
            cv2.rectangle(img, tuple(a[0:2].astype(int)), tuple(a[2:].astype(int)), self.colormap[int(p[i])])
        skimage.io.imshow(img)
        skimage.io.show()

    def center_to_corner(self, true_box, to_all_scale=True):
        if to_all_scale:
            x1 = (true_box[:, 0:1]-true_box[:, 2:3]/2)*self.in_w
            y1 = (true_box[:, 1:2]-true_box[:, 3:4]/2)*self.in_h
            x2 = (true_box[:, 0:1]+true_box[:, 2:3]/2)*self.in_w
            y2 = (true_box[:, 1:2]+true_box[:, 3:4]/2)*self.in_h
        else:
            x1 = (true_box[:, 0:1]-true_box[:, 2:3]/2)
            y1 = (true_box[:, 1:2]-true_box[:, 3:4]/2)
            x2 = (true_box[:, 0:1]+true_box[:, 2:3]/2)
            y2 = (true_box[:, 1:2]+true_box[:, 3:4]/2)

        xyxy_box = np.hstack([x1, y1, x2, y2])
        return xyxy_box

    def corner_to_center(self, xyxy_box, from_all_scale=True):
        if from_all_scale:
            x = ((xyxy_box[:, 2:3]-xyxy_box[:, 0:1])/2+xyxy_box[:, 0:1])/self.in_w
            y = ((xyxy_box[:, 3:4]-xyxy_box[:, 1:2])/2+xyxy_box[:, 1:2])/self.in_h
            w = (xyxy_box[:, 2:3]-xyxy_box[:, 0:1])/self.in_w
            h = (xyxy_box[:, 3:4]-xyxy_box[:, 1:2])/self.in_h
        else:
            x = ((xyxy_box[:, 2:3]-xyxy_box[:, 0:1])/2+xyxy_box[:, 0:1])
            y = ((xyxy_box[:, 3:4]-xyxy_box[:, 1:2])/2+xyxy_box[:, 1:2])
            w = (xyxy_box[:, 2:3]-xyxy_box[:, 0:1])
            h = (xyxy_box[:, 3:4]-xyxy_box[:, 1:2])

        true_box = np.hstack([x, y, w, h])
        return true_box


def tf_xywh_to_all(xy: tf.Tensor, wh: tf.Tensor, helper: Helper):
    """ rescale the xy wh to [0~1]

    Parameters
    ----------
    xy : tf.Tensor

    wh : tf.Tensor

    handler : DataHandler

    Returns
    -------
    tuple
        after process xy wh
    """
    xy_A = xy[..., :]*np.array([helper.grid_w, helper.grid_h])+helper.xy_offset
    wh_A = wh[..., :]*helper.wh_scale

    return xy_A, wh_A


def tf_reshape_box(true_xy_A: tf.Tensor, true_wh_A: tf.Tensor, p_xy_A: tf.Tensor, p_wh_A: tf.Tensor, helper: Helper)->tuple:
    """ reshape the xywh to [?,h,w,anchor_nums,true_box_nums,2]
    NOTE  must use obj mask in atrue xywh !
    Parameters
    ----------
    true_xy_A : tf.Tensor
        shape will be [true_box_nums,2]
    true_wh_A : tf.Tensor
        shape will be [true_box_nums,2]
    p_xy_A : tf.Tensor
        shape will be [?,h,w,anhor_nums,2]
    p_wh_A : tf.Tensor
        shape will be [?,h,w,anhor_nums,2]


    Returns
    -------
    tuple
        after reshape xywh
    """
    true_cent = true_xy_A[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, ...]
    true_box_wh = true_wh_A[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, ...]

    true_cent = tf.tile(true_cent, [helper.batch_size, helper.out_h, helper.out_w, len(helper.anchors), 1, 1])
    true_box_wh = tf.tile(true_box_wh, [helper.batch_size, helper.out_h, helper.out_w, len(helper.anchors), 1, 1])

    pred_cent = p_xy_A[..., tf.newaxis, :]
    pred_box_wh = p_wh_A[..., tf.newaxis, :]
    pred_cent = tf.tile(pred_cent, [1, 1, 1, 1, tf.shape(true_xy_A)[0], 1])
    pred_box_wh = tf.tile(pred_box_wh, [1, 1, 1, 1, tf.shape(true_wh_A)[0], 1])

    return true_cent, true_box_wh, pred_cent, pred_box_wh


def tf_iou(cent1: tf.Tensor, box1: tf.Tensor,  cent2: tf.Tensor, box2: tf.Tensor)->tf.Tensor:
    """ use tensorflow calc iou NOTE shape will be [?,h,w,num_anchor,true_box_num,2]

    Parameters
    ----------
    cent1 : tf.Tensor
        (x,y) 
    box1 : tf.Tensor
        (w,h)
    cent2 : tf.Tensor
        (x,y)
    box2 : tf.Tensor
        (w,h)

    Returns
    -------
    tf.Tensor
        iou score tensor 
        NOTE shape will be [?,h,w,num anchor,true box num,1]
    """

    box1_half_wh = box1[..., :]/2.
    box1_left_top = cent1[..., :]-box1_half_wh
    box1_right_bottom = cent1[..., :]+box1_half_wh

    box2_half_wh = box2[..., :]/2.
    box2_left_top = cent2[..., :]-box2_half_wh
    box2_right_bottom = cent2[..., :]+box2_half_wh

    intersect_wh = tf.maximum(tf.minimum(box1_right_bottom[..., :], box2_right_bottom[..., :]) - tf.maximum(box1_left_top, box2_left_top), [0., 0.])

    intersect_area = intersect_wh[..., 0]*intersect_wh[..., 1]

    box1_area = box1[..., 0]*box1[..., 1]
    box2_area = box2[..., 0]*box2[..., 1]

    return intersect_area/(box1_area+box2_area-intersect_area)
