import tensorflow as tf
from core import common

slim = tf.contrib.slim

class darnet53(object):
    """
    define darnet53 network 
    """


    def __init__(self, inputs):
      self.outputs = self.forward(inputs) 
    

    def _darknet_block(self, inputs, filters):
        shortcut = inputs
        inputs = common._conv2d_fixed_padding(inputs, filters * 1, 1)
        inputs = common._conv2d_fixed_padding(inputs, filters * 2, 3)

        inputs = inputs + shortcut
        return inputs


    def forward(self, inputs):
        # inputs [none, 416, 416, 3]
        inputs = common._conv2d_fixed_padding(inputs, 32, 3, strides=1)
        inputs = common._conv2d_fixed_padding(inputs, 64, 3, strides=2)
        inputs = self._darknet_block(inputs, 32)
        inputs = common._conv2d_fixed_padding(inputs, 128, 3, strides=2)

        for _ in range(2):
            inputs = self._darknet_block(inputs, 64)
        inputs = common._conv2d_fixed_padding(inputs, 256, 3, strides=2)
        
        for _ in range(8):
            inputs = self._darknet_block(inputs, 128)
        route_3 = inputs
        inputs = common._conv2d_fixed_padding(inputs, 512, 3, strides=2)

        for _ in range(8):
            inputs = self._darknet_block(inputs, 256)
        route_2 = inputs
        inputs = common._conv2d_fixed_padding(inputs, 1024, 3,strides=2)

        for _ in range(4):
            inputs = self._darknet_block(inputs, 512)
        route_1 = inputs

        return route_1, route_2, route_3

class yolov3(object):

    
    def __init__(self, num_classes, anchors, batch_norm_decay=0.9, leaky_relu=0.1):
        self.num_classes = num_classes
        self.anchors = anchors
        self.batch_norm_decay = batch_norm_decay
        self.leaky_relu = leaky_relu
        self.feature_maps = []
        #[none, 13, 13 255], [none, 26, 26, 255], [none, 52, 52, 255]
    
    def _yolo_block(self, inputs, filters):
        inputs = common._conv2d_fixed_padding(inputs, filters * 1, 1)
        inputs = common._conv2d_fixed_padding(inputs, filters * 2, 3)
        inputs = common._conv2d_fixed_padding(inputs, filters * 1, 1)
        inputs = common._conv2d_fixed_padding(inputs, filters * 2, 3)
        inputs = common._conv2d_fixed_padding(inputs, filters * 1, 1)
        route = inputs
        inputs = common._conv2d_fixed_padding(inputs, filters * 2, 3)
        return route, inputs

    
    def _detection_layer(self, inputs, anchors):
        num_anchors = len(anchors)
        feature_maps = slim.conv2d(inputs, num_anchors * (5 + self.num_classes),
                                    1,
                                    stride=1,
                                    normalizer_fn=None,
                                    activation_fn=None,
                                    biases_initializer=tf.zeros_initializer())
        return feature_maps


    @staticmethod
    def _upsample(inputs, out_shape):
        new_height, new_width = out_shape[1], out_shape[2]
        inputs = tf.image.resize_nearest_neighbor(inputs, (new_height, new_width))
        inputs = tf.identity(inputs, name='upsample')

    
    def forward(self, inputs, is_training=False, reuse=False):
        """
        creates yolo v3 model

        inputs [none, height, width, channels0]
        outputs: [none, 13, 13, 75], [none, 26, 26, 75], [none, 52, 52, 75]
        """
        #[height, width]
        self.img_size = tf.shape(inputs)[1:3]

        batch_norm_params = {
            'decay': self.batch_norm_decay,
            'epsilon': 1e-05,
            'scale':True,
            'is_training': is_training
        }

        with slim.arg_scope([slim.conv2d, slim.batch_norm, common._fixed_padding], reuse=reuse):
            with slim.arg_scope([slim.conv2d],
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                biases_initializer=None,
                                activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=self.leaky_relu)):
                with tf.variable_scope('darknet-53'):
                    route_1, route_2, route_3 = darnet53(inputs).outputs
                
                with tf.variable_scope('yolo-v3'):
                    route, inputs = self._yolo_block(route_1, 512)
                    feature_maps_1 = self._detection_layer(inputs, self.anchors[6:9])
                    feature_maps_1 = tf.identity(feature_maps, name='feature_maps_1')

                    inputs = common._conv2d_fixed_padding()
                    upsample_size = route_2.get_shape().as_list()
                    inputs = self._upsample(inputs, upsample_size)
                    inputs = tf.concat([inputs, route_2], axis=3)

                    route, inputs = self._yolo_block(inputs, 256)
                    feature_maps_2 = self._detection_layer(inputs, self.anchors[3:6])
                    feature_maps_2 = tf.identity(feature_maps_2, name='feature_maps_2')

                    inputs = common._conv2d_fixed_padding(route, 128, 1)
                    upsample_size = route_3.get_shape().as_list()
                    inputs = self._upsample(inputs, upsample_size)
                    inputs = tf.concat([inputs, route_3], axis=3)

                    route, inputs = self._yolo_block(inputs, 128)
                    feature_maps_3 = self._detection_layer(inputs, self.anchors[0:3])
                return feature_maps_1, feature_maps_2, feature_maps_3

    def _reorg_layer(self, feature_maps, anchors):

        """
        input : [none, 13, 13, 75]
        """
        num_anchors = len(anchors)# 3
        gird_size = feature_maps.shape.as_list()[1:3] #[13, 13]
        #[height, width] // [13, 13]
        stride = tf.cast(self.img_size // gird_size, tf.float32)
        #reshpe [none, 13, 13, num_anchors, 5(b_boxes) + num_classes]
        #5(b_boxes) =  box_centers, box_size, con_logits, pro_logits
        feature_maps = tf.reshape(feature_maps,
                                  [-1, gird_size[0], gird_size[1], num_anchors, 5 + self.num_classes])
        box_centers, box_sizes, conf_logits, prob_logits = tf.split(
            feature_maps, [2, 2, 1, self.num_classes], axis=1)
        
        box_centers = tf.nn.sigmoid(box_centers)

        grid_x = tf.range(gird_size[1], dtype=tf.int32)
        grid_y = tf.range(gird_size[0], dtype=tf.int32)

        a,b = tf.meshgrid(grid_x, grid_y)

        x_offest = tf.reshape(a, (-1, 1))
        y_offest = tf.reshape(b, (-1, 1))
        x_y_offset = tf.concat([x_offest, y_offest], axis=-1)
        x_y_offset = tf.reshape(x_y_offset, [gird_size[0], gird_size[1], 1, 2])
        x_y_offset = tf.cast(x_y_offset, tf.float32)

        box_centers = box_centers + x_y_offset
        box_centers = box_centers * stride[::1]

        box_sizes = tf.exp(box_sizes) * anchors
        boxes = tf.concat([box_centers, box_sizes], axis=-1)
        return x_y_offset, boxes, conf_logits, prob_logits

    def _reshape(self, x_y_offset, boxes, confs, probs):
        gird_size = 

    def predict(self, feature_maps):
        """
        input feature maps  [none, 13, 13, 255]
                            [none, 26, 26, 255]
                            [none, 52, 52, 255]
        """
        feature_maps_1 , feature_maps_2 , feature_maps_3 = feature_maps
        feature_maps_anchors = [(feature_maps_1, self.anchors[6:9]),
                                (feature_maps_2, self.anchors[3:6]),
                                (feature_maps_3, self.anchors[0:3])]
        
        results = [self._reorg_layer(feature_maps, anchors) for (feature_maps, anchors) in feature_maps_anchors]
        boxes_list, confs_list, probs_list = [], [], []

        for result in results:
            boxes, confs_logits, prob_logits = self._reshape(*result)
            

            
