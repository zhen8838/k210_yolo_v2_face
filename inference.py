import tensorflow as tf
from tools.utils import Helper, tf_xywh_to_all
import skimage
import numpy as np
from scipy.special import expit
import sys
import argparse


def tf_center_to_corner(box):
    x1 = (box[..., 0:1]-box[..., 2:3]/2)
    y1 = (box[..., 1:2]-box[..., 3:4]/2)
    x2 = (box[..., 0:1]+box[..., 2:3]/2)
    y2 = (box[..., 1:2]+box[..., 3:4]/2)
    box = tf.concat([y1, x1, y2, x2], -1)
    return box


def main(pb_path, class_num, anchor_file, image_size, image_path):
    g = tf.get_default_graph()
    helper = Helper(None, None, class_num, anchor_file, image_size, (7, 10))

    test_img = helper._read_img(image_path, True)
    test_img = helper._process_img(test_img, None, is_training=False)[0]

    with tf.gfile.GFile(pb_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

    inputs = g.get_tensor_by_name('Input_image:0')
    pred_label = g.get_tensor_by_name('Yolo/Final/conv2d/BiasAdd:0')
    """ reshape the model output """
    pred_label = tf.reshape(pred_label, [-1, helper.out_h, helper.out_w, len(helper.anchors), 5+class_num])

    """ split the label """
    pred_xy = pred_label[..., 0:2]
    pred_wh = pred_label[..., 2:4]
    pred_confidence = pred_label[..., 4:5]
    pred_cls = pred_label[..., 5:]

    pred_xy = tf.nn.sigmoid(pred_xy)
    pred_wh = tf.exp(pred_wh)
    pred_confidence_sigmoid = tf.nn.sigmoid(pred_confidence)
    obj_mask = pred_confidence_sigmoid[..., 0] > .7
    """ reshape box  """
    pred_xy_A, pred_wh_A = tf_xywh_to_all(pred_xy, pred_wh, helper)

    box = tf.concat([pred_xy_A, pred_wh_A], -1)

    yxyx_box = tf_center_to_corner(box)
    yxyx_box = tf.boolean_mask(yxyx_box, obj_mask)
    """ nms  """
    select = tf.image.non_max_suppression(yxyx_box,
                                          scores=tf.reshape(tf.boolean_mask(pred_confidence_sigmoid, obj_mask), (-1, )),
                                          max_output_size=30)
    vaild_box = tf.gather(yxyx_box, select)
    vaild_box = vaild_box[tf.newaxis, :, :]
    """ draw box """
    img_box = tf.image.draw_bounding_boxes(inputs, vaild_box)

    """ run """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # todo -----
        test_img = helper._read_img(image_path, is_resize=True)
        test_img, _ = helper._process_img(test_img, None, False)
        test_img = test_img[np.newaxis, :, :, :]
        img_box_, vaild_box_, yxyx_box_, pred_xy_, pred_wh_, pred_confidence_sigmoid_ = sess.run(
            [img_box, vaild_box, yxyx_box, pred_xy, pred_wh, pred_confidence_sigmoid], feed_dict={inputs: test_img})
    skimage.io.imshow(img_box_[0])
    skimage.io.show()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--pb_path', type=str, help='pb file path', default='Freeze_save.pb')
    parser.add_argument('--class_num', type=int, help='trian class num', default=1)
    parser.add_argument('--anchor_file', type=str, help='anchors list file ', default='data/anchors.list')
    parser.add_argument('--image_size', type=int, help='net work input image size', default=(240, 320), nargs='+')
    parser.add_argument('--image_path', type=str, help='the face image', default='data/2.jpg')
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])
    # pb_path, class_num, anchor_file, image_size, image_path='Freeze_save.pb',1,'data/anchors.list',(240,320),'data/2.jpg'
    main(args.pb_path, args.class_num, args.anchor_file, args.image_size, args.image_path)
