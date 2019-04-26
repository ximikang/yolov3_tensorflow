import tensorflow as tf
from core import common
slim = tf.contrib.slim


class darknet53(object):


    def __init__(self, inputs):
        self.outputs = self.forward(inputs)


    def _darknet_block(self, inputs, filters):
        shortcut = inputs
        inputs = common.conv2d_fixed_padding(inputs, filters*1, 1)
        inputs = common.conv2d_fixed_padding(inputs, filters*3, 3)

        inputs = inputs + shortcut
        return inputs


    def forward(self, inputs):
        inputs = common.conv2d_fixed_padding(inputs, 32, 3, strides=1)
        inputs = common.conv2d_fixed_padding(inputs, 64, 3, strides=2)
        inputs = self._darknet_block(inputs, 32)
        inputs = common.conv2d_fixed_padding(inputs, 128, 3, strides=2)

        for i in range(2):
            inputs = self._darknet_block(inputs, 64)
        inputs = common.conv2d_fixed_padding(inputs, 256, 3, strides=2)

        for i in range(8):
            inputs = self._darknet_block(inputs, 128)
        route_1 = inputs
        inputs = common.conv2d_fixed_padding(inputs, 512, 3, strides=2)

        for i in range(8):
            inputs = self._darknet_block(inputs, 256)
        route_2 = inputs
        inputs = common.conv2d_fixed_padding(inputs, 1024, 3, strides=2)

        for i in range(4):
            inputs = self._darknet_block(inputs, 512)
        route_3 = inputs

        return route_1, route_2, route_3


class yolov3(object):


    def __init__(self, num_classes, anchors, batch_norm_decay=0.9, leaky_relu=0.1):
        self._ANCHORS = anchors
        self._BATCH_NORM_DECAY = 0.9
        self._NUM_CLASSES = num_classes
        self._LEAKY_RELU = 0.1
        self.feature_maps = []

    def _yolo_block(self, inputs, filters):
        inputs = common.conv2d_fixed_padding(inputs, filters * 1, 1)
        inputs = common.conv2d_fixed_padding(inputs, filters * 2, 3)
        inputs = common.conv2d_fixed_padding(inputs, filters * 1, 1)
        inputs = common.conv2d_fixed_padding(inputs, filters * 2, 3)
        inputs = common.conv2d_fixed_padding(inputs, filters * 1, 1)
        route = inputs
        inputs = common.conv2d_fixed_padding(inputs, filters * 2, 3)
        return route, inputs

    def forward(self, inputs, is_training=False, reuse=False):
        self.image_size = tf.shape(inputs)[1:3]

        batch_norm_params = {
            'decay': self._BATCH_NORM_DECAY,
            'epsilon': 1e-05,
            'scale': True,
            'is_training': is_training,
            'fused': None,}

        #set activate function and parameters for conv2d and batch_normlization
        with slim.arg_scope([slim.conv2d, slim.batch_norm, common._fixed_padding], reuse=reuse):
            with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
                       normalizer_params=batch_norm_params,
                       biases_initializer=None,
                       activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=self._LEAKY_RELU)):
                with tf.variable_scope('darknet-53'):
                    route_1, route_2, route_3 = darknet53(inputs).outputs
                with tf.variable_scope('yolov3'):
                    route, inputs = self._yolo_block(inputs, 512)
                    feature_map_1 = self.detection_layer(inputs, self._ANCHORS[6:9])
                    feature_map_1 = tf.identity(feature_map_1, name='feature_map_1')

                    inputs = common.conv2d_fixed_padding(route, 256, 1)
                    upsample_size = route_2.get_shape().as_list()
                    inputs = self._upsample(inputs, upsample_size)
                    inputs = tf.concat([inputs, route_2], axis=3)

                    route, inputs = self._yolo_block(inputs, 256)
                    feature_map_2 = self.detection_layer(inputs, self._ANCHORS[3:6])
                    feature_map_2 = tf.identity(feature_map_2, name='feature_map_2')

                    inputs = common.conv2d_fixed_padding(route, 128, 1)
                    upsample_size = route_1.get_shape().as_list()
                    inputs = self._upsample(inputs, upsample_size)
                    inputs = tf.concat([inputs, route_1], axis=3)

                    route, inputs = self._yolo_block(inputs, 128)
                    feature_map_3 = self.detection_layer(inputs, self._ANCHORS[0:3])
                    feature_map_3 = tf.identity(feature_map_3, name='feature_map_3')

                return feature_map_1, feature_map_2, feature_map_3