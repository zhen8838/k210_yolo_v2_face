import tensorflow as tf
from tools.utils import Helper, restore_ckpt, tf_reshape_box, tf_iou, tf_xywh_to_all, write_arguments_to_file
from models.yolonet import pureconv, mobile_yolo
from tensorflow.contrib import slim
import os
from datetime import datetime
from tqdm import tqdm
import numpy as np
import sys
import argparse
import importlib


def calc_noobj_mask(t_xy, t_wh, p_xy, p_wh, obj_mask, iou_thresh, helper: Helper):
    t_xy_A, t_wh_A = tf_xywh_to_all(t_xy, t_wh, helper)
    p_xy_A, p_wh_A = tf_xywh_to_all(p_xy, p_wh, helper)

    t_xy_A = tf.boolean_mask(t_xy_A, obj_mask)
    t_wh_A = tf.boolean_mask(t_wh_A, obj_mask)

    t_cent, t_box_wh, p_cent, p_box_wh = tf_reshape_box(t_xy_A, t_wh_A, p_xy_A, p_wh_A, helper)
    iou_score = tf_iou(t_cent, t_box_wh, p_cent, p_box_wh)
    iou_score = tf.reduce_max(iou_score, axis=-1, keepdims=True)
    iou_mask = iou_score[..., 0] > iou_thresh

    noobj_mask = tf.logical_not(tf.logical_or(obj_mask, iou_mask))

    return noobj_mask


def main(args,
         train_set,
         class_num,
         train_classifier,
         pre_ckpt,
         model_def,
         is_augmenter,
         anchor_file,
         image_size,
         output_size,
         batch_size,
         rand_seed,
         max_nrof_epochs,
         init_learning_rate,
         learning_rate_decay_epochs,
         learning_rate_decay_factor,
         obj_weight,
         noobj_weight,
         obj_thresh,
         iou_thresh,
         log_dir):
    g = tf.get_default_graph()
    tf.set_random_seed(rand_seed)
    """ import network """
    network = eval(model_def)

    """ generate the dataset """
    # [(0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434), (7.88282, 3.52778), (9.77052, 9.16828)]
    helper = Helper('data/{}_img.list'.format(train_set), 'data/{}_ann.list'.format(train_set), class_num, anchor_file, image_size, output_size)
    helper.set_dataset(batch_size, rand_seed, is_training=(is_augmenter == 'True'))
    next_img, next_label = helper.get_iter()

    """ define the model """
    batch_image = tf.placeholder_with_default(next_img, shape=[None, image_size[0], image_size[1], 3], name='Input_image')
    batch_label = tf.placeholder_with_default(next_label, shape=[None, output_size[0], output_size[1], len(helper.anchors), 5+class_num], name='Input_label')
    training_control = tf.placeholder_with_default(True, shape=[], name='training_control')
    true_label = tf.identity(batch_label)
    nets, endpoints = network(batch_image, len(helper.anchors), class_num, phase_train=training_control)

    """ reshape the model output """
    pred_label = tf.reshape(nets, [-1, output_size[0], output_size[1], len(helper.anchors), 5+class_num], name='predict')
    """ split the label """
    pred_xy = pred_label[..., 0:2]
    pred_wh = pred_label[..., 2:4]
    pred_confidence = pred_label[..., 4:5]
    pred_cls = pred_label[..., 5:]

    pred_xy = tf.nn.sigmoid(pred_xy)
    pred_wh = tf.exp(pred_wh)
    pred_confidence_sigmoid = tf.nn.sigmoid(pred_confidence)

    true_xy = true_label[..., 0:2]
    true_wh = true_label[..., 2:4]
    true_confidence = true_label[..., 4:5]
    true_cls = true_label[..., 5:]

    obj_mask = true_confidence[..., 0] > obj_thresh

    """ calc the noobj mask ~ """
    if train_classifier == 'True':
        noobj_mask = tf.logical_not(obj_mask)
    else:
        noobj_mask = calc_noobj_mask(true_xy, true_wh, pred_xy, pred_wh, obj_mask, iou_thresh=iou_thresh, helper=helper)

    """ define loss """
    xy_loss = tf.reduce_sum(tf.square(tf.boolean_mask(true_xy, obj_mask)-tf.boolean_mask(pred_xy, obj_mask)))/batch_size
    wh_loss = tf.reduce_sum(tf.square(tf.boolean_mask(true_wh, obj_mask)-tf.boolean_mask(pred_wh, obj_mask)))/batch_size
    obj_loss = obj_weight * tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.boolean_mask(true_confidence, obj_mask), logits=tf.boolean_mask(pred_confidence, obj_mask)))/batch_size
    noobj_loss = noobj_weight * tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.boolean_mask(true_confidence,
                                                                                                             noobj_mask), logits=tf.boolean_mask(pred_confidence, noobj_mask)))/batch_size
    cls_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.boolean_mask(true_cls, obj_mask), logits=tf.boolean_mask(pred_cls, obj_mask)))/batch_size

    # xy_loss = tf.losses.mean_squared_error(tf.boolean_mask(true_xy, obj_mask), tf.boolean_mask(pred_xy, obj_mask))
    # wh_loss = tf.losses.mean_squared_error(tf.boolean_mask(true_wh, obj_mask), tf.boolean_mask(pred_wh, obj_mask))
    # obj_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.boolean_mask(true_confidence, obj_mask), logits=tf.boolean_mask(pred_confidence, obj_mask), weights=5.0)
    # noobj_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.boolean_mask(true_confidence, noobj_mask), logits=tf.boolean_mask(pred_confidence, noobj_mask), weights=.5)
    # cls_loss = tf.losses.softmax_cross_entropy(onehot_labels=tf.boolean_mask(true_cls, obj_mask), logits=tf.boolean_mask(pred_cls, obj_mask))

    if train_classifier == 'True':
        total_loss = obj_loss+noobj_loss+cls_loss
    else:
        total_loss = obj_loss+noobj_loss+cls_loss+xy_loss+wh_loss

    """ define steps """
    global_steps = tf.train.create_global_step()

    """ define learing rate """
    current_learning_rate = tf.train.exponential_decay(init_learning_rate, global_steps, helper.epoch_step // learning_rate_decay_epochs,
                                                       learning_rate_decay_factor, staircase=False)
    """ define train_op """
    train_op = slim.learning.create_train_op(total_loss, tf.train.AdamOptimizer(current_learning_rate), global_steps)

    """ calc the accuracy """
    precision, prec_op = tf.metrics.precision_at_thresholds(true_confidence, pred_confidence_sigmoid, [obj_thresh])
    test_precision, test_prec_op = tf.metrics.precision_at_thresholds(true_confidence, pred_confidence_sigmoid, [obj_thresh])
    recall, recall_op = tf.metrics.recall_at_thresholds(true_confidence, pred_confidence_sigmoid, [obj_thresh])
    test_recall, test_recall_op = tf.metrics.recall_at_thresholds(true_confidence, pred_confidence_sigmoid, [obj_thresh])
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        """ must save the bn paramter! """
        var_list = tf.global_variables()+tf.local_variables()  # list(set(tf.trainable_variables() + [g for g in tf.global_variables() if 'moving_' in g.name]))
        saver = tf.train.Saver(var_list)

        # init the model and restore the pre-train weight
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())  # NOTE the accuracy must init local variable
        restore_ckpt(sess, var_list, pre_ckpt)
        # define the log and saver
        subdir = os.path.join(log_dir, datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S'))
        train_writer = tf.summary.FileWriter(subdir, graph=sess.graph)
        write_arguments_to_file(args, os.path.join(subdir, 'arguments.txt'))
        tf.summary.scalar('total_loss', total_loss)
        tf.summary.scalar('obj_loss', obj_loss)
        tf.summary.scalar('noobj_loss', noobj_loss)
        tf.summary.scalar('mse_loss', xy_loss+wh_loss)
        tf.summary.scalar('class_loss', cls_loss)
        tf.summary.scalar('leraning_rate', current_learning_rate)
        tf.summary.scalar('precision', precision[0])
        tf.summary.scalar('recall', recall[0])
        merged = tf.summary.merge_all()
        t_prec_summary = tf.summary.scalar('test_precision', test_precision[0])
        t_recall_summary = tf.summary.scalar('test_recall', test_recall[0])

        try:
            for i in range(max_nrof_epochs):
                with tqdm(total=helper.epoch_step, bar_format='{n_fmt}/{total_fmt} |{bar}| {rate_fmt}{postfix}', unit=' batch', dynamic_ncols=True) as t:
                    for j in range(helper.epoch_step):
                        if j % 30 == 0:
                            summary1, summary2, _, _, step_cnt = sess.run(
                                [t_prec_summary, t_recall_summary, test_recall_op, test_prec_op, global_steps], feed_dict={training_control: False})
                            train_writer.add_summary(summary1, step_cnt)
                            train_writer.add_summary(summary2, step_cnt)
                        else:
                            summary, _, total_l, prec, _, _,  lr, step_cnt = sess.run(
                                [merged, train_op, total_loss, precision, prec_op, recall_op, current_learning_rate, global_steps])
                            t.set_postfix(loss='{:<5.3f}'.format(total_l), prec='{:<4.2f}%'.format(prec[0]*100), lr='{:f}'.format(lr))
                            train_writer.add_summary(summary, step_cnt)
                        t.update()
            saver.save(sess, save_path=os.path.join(subdir, 'model.ckpt'), global_step=global_steps)
            print('save over')
        except KeyboardInterrupt as e:
            saver.save(sess, save_path=os.path.join(subdir, 'model.ckpt'), global_step=global_steps)
            print('save over')


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_set',                  type=str,   help='trian file lists',                choices=['voc', 'coco', 'fddb'])
    parser.add_argument('--class_num',                  type=int,   help='trian class num',                 default=20)
    parser.add_argument('--train_classifier',           type=str,   help='wether train the classsifier',    choices=['True', 'False'], default='False')
    parser.add_argument('--pre_ckpt',                   type=str,   help='pre-train ckpt dir',              default='None')
    parser.add_argument('--model_def',                  type=str,   help='Model definition.',               choices=['mobile_yolo', 'pureconv'], default=pureconv)
    parser.add_argument('--augmenter',                  type=str,   help='use image augmenter',             choices=['True', 'False'], default='True')
    parser.add_argument('--anchor_file',                type=str,   help='anchors list file ',              default='data/anchors.list')
    parser.add_argument('--image_size',                 type=int,   help='net work input image size',       default=(240, 320), nargs='+')
    parser.add_argument('--output_size',                type=int,   help='net work output image size',      default=(7, 10), nargs='+')
    parser.add_argument('--batch_size',                 type=int,   help='batch size',                      default=32)
    parser.add_argument('--rand_seed',                  type=int,   help='random seed',                     default=6)
    parser.add_argument('--max_nrof_epochs',            type=int,   help='epoch num',                       default=10)
    parser.add_argument('--init_learning_rate',         type=float, help='init learing rate',               default=0.0005)
    parser.add_argument('--learning_rate_decay_epochs', type=int,   help='learning rate decay epochs',      default=10)
    parser.add_argument('--learning_rate_decay_factor', type=float, help='learning rate decay factor',      default=1.0)
    parser.add_argument('--obj_weight',                 type=float, help='obj loss weight',                 default=5.0)
    parser.add_argument('--noobj_weight',               type=float, help='noobj loss weight',               default=0.5)
    parser.add_argument('--obj_thresh',                 type=float, help='obj mask thresh',                 default=0.75)
    parser.add_argument('--iou_thresh',                 type=float, help='iou mask thresh',                 default=0.5)
    parser.add_argument('--log_dir',                    type=str,   help='log dir',                         default='log')

    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])
    main(args,
         args.train_set,
         args.class_num,
         args.train_classifier,
         args.pre_ckpt,
         args.model_def,
         args.augmenter,
         args.anchor_file,
         args.image_size,
         args.output_size,
         args.batch_size,
         args.rand_seed,
         args.max_nrof_epochs,
         args.init_learning_rate,
         args.learning_rate_decay_epochs,
         args.learning_rate_decay_factor,
         args.obj_weight,
         args.noobj_weight,
         args.obj_thresh,
         args.iou_thresh,
         args.log_dir)
