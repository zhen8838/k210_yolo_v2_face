import tensorflow as tf
from tensorflow.contrib import slim
from models.mobilenet_v1 import mobilenet_v1_arg_scope, mobilenet_v1_base


def mobile_yolo(images: tf.Tensor, anchor_num, class_num, phase_train: bool):
    with slim.arg_scope(mobilenet_v1_arg_scope(is_training=phase_train)):
        nets, endpoints = mobilenet_v1_base(images, depth_multiplier=0.5)

    # add the new layer
    with tf.variable_scope('Yolo'):
        with slim.arg_scope([slim.batch_norm], scale=True, is_training=phase_train):
            nets = slim.separable_conv2d(nets, None, (3, 3), activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm, scope='Conv2d_0_depthwise')
            nets = slim.conv2d(nets, 256, (1, 1), activation_fn=tf.nn.leaky_relu, normalizer_fn=slim.batch_norm, scope='Conv2d_0_pointwise')
            # nets=[7,10,256]
            
            nets = slim.conv2d(nets, 128, (3, 3), activation_fn=tf.nn.leaky_relu, normalizer_fn=slim.batch_norm, scope='Conv2d_1')
            # nets=[7,10,128]

            nets = slim.conv2d(nets, 64, (3, 3), activation_fn=tf.nn.leaky_relu, normalizer_fn=slim.batch_norm, scope='Conv2d_2')
            # nets=[7,10,64]
        with tf.variable_scope('Final'):
            nets = tf.layers.conv2d(nets, anchor_num*(5+class_num), (3, 3), padding='same')
    return nets, endpoints


def pureconv(images: tf.Tensor, anchor_num, class_num, phase_train: bool):
    """ this network input should be 240*320 """
    with tf.variable_scope('Yolo'):
        with tf.variable_scope('convd_1'):
            nets = tf.layers.conv2d(images, 16, (3, 3), padding='same')
            nets = tf.layers.batch_normalization(nets, training=phase_train)
            nets = tf.nn.leaky_relu(nets)
            nets = tf.layers.max_pooling2d(nets, pool_size=(2, 2), strides=(2, 2))
            # nets=[120,160,16]
        with tf.variable_scope('convd_2'):
            nets = tf.layers.conv2d(nets, 32, (3, 3), padding='same')
            nets = tf.layers.batch_normalization(nets, training=phase_train)
            nets = tf.nn.leaky_relu(nets)
            nets = tf.layers.max_pooling2d(nets, pool_size=(2, 2), strides=(2, 2))
            # nets=[60,80,16]
        with tf.variable_scope('convd_3'):
            nets = tf.layers.conv2d(nets, 64, (3, 3), padding='same')
            nets = tf.layers.batch_normalization(nets, training=phase_train)
            nets = tf.nn.leaky_relu(nets)
            nets = tf.layers.max_pooling2d(nets, pool_size=(2, 2), strides=(2, 2))
            # nets=[30,40,64]
        with tf.variable_scope('convd_4'):
            nets = tf.layers.conv2d(nets, 64, (3, 3), padding='same')
            nets = tf.layers.batch_normalization(nets, training=phase_train)
            nets = tf.nn.leaky_relu(nets)
            # nets=[30,40,64]
        with tf.variable_scope('convd_5'):
            nets = tf.layers.conv2d(nets, 128, (3, 3), padding='same')
            nets = tf.layers.batch_normalization(nets, training=phase_train)
            nets = tf.nn.leaky_relu(nets)
            nets = tf.layers.max_pooling2d(nets, pool_size=(2, 2), strides=(2, 2))
            # nets=[15,20,128]
        with tf.variable_scope('convd_6'):
            nets = tf.layers.conv2d(nets, 256, (3, 3), padding='same')
            nets = tf.layers.batch_normalization(nets, training=phase_train)
            nets = tf.nn.leaky_relu(nets)
            nets = tf.layers.max_pooling2d(nets, pool_size=(2, 2), strides=(2, 2))
            # nets=[7,10,256]
        with tf.variable_scope('convd_7'):
            nets = tf.layers.conv2d(nets, 128, (3, 3), padding='same')
            nets = tf.layers.batch_normalization(nets, training=phase_train)
            nets = tf.nn.leaky_relu(nets)
            # nets=[7,10,128]
        with tf.variable_scope('convd_8'):
            nets = tf.layers.conv2d(nets, 128, (3, 3), padding='same')
            nets = tf.layers.batch_normalization(nets, training=phase_train)
            nets = tf.nn.leaky_relu(nets)
            # nets=[7,10,128]
        with tf.variable_scope('Final'):
            nets = tf.layers.conv2d(nets, anchor_num*(5+class_num), (3, 3), padding='same')
        # nets=[7,10,5]
        endpoints = None
    return nets, endpoints
