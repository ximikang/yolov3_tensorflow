import tensorflow as tf
from core import common
slim = tf.contrib.slim


class darknet53(object):


    def __init__(self, inputs):
        self.outputs = self.forward(inputs)

    @staticmethod
    def _darknet_block(inputs, filters):
        shortcut = inputs
        inputs = common.conv2d_fixed_padding(inputs, filters*1, 1)
        inputs = common.conv2d_fixed_padding(inputs, filters*2, 3)

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

    @staticmethod
    def _yolo_block(inputs, filters):
        inputs = common.conv2d_fixed_padding(inputs, filters * 1, 1)
        inputs = common.conv2d_fixed_padding(inputs, filters * 2, 3)
        inputs = common.conv2d_fixed_padding(inputs, filters * 1, 1)
        inputs = common.conv2d_fixed_padding(inputs, filters * 2, 3)
        inputs = common.conv2d_fixed_padding(inputs, filters * 1, 1)
        route = inputs
        inputs = common.conv2d_fixed_padding(inputs, filters * 2, 3)
        return route, inputs


    def _detection_layer(self, inputs, anchors):
        num_anchors = len(anchors)
        feature_map = slim.conv2d(inputs, num_anchors * (5 + self._NUM_CLASSES), 1,
                                  stride=1, normalizer_fn=None,
                                  activation_fn=None,
                                  biases_initializer=tf.zeros_initializer())
        return feature_map


    @staticmethod
    def _upsample(inputs, out_shape):
        """
        upsample layer
        :param inputs: inputs tensor
        :param out_shape: output shape
        :return: upsample with nearest neighbor
        """
        new_height, new_width = out_shape[1], out_shape[2]
        inputs = tf.image.resize_nearest_neighbor(inputs, (new_height, new_width))
        inputs = tf.identity(inputs, name='upsampled')
        return inputs


    def forward(self, inputs, is_training=False, reuse=False):
        """
            input: 416*416*3
            outpus: [13*13*255], [26*26*255], [52*52*255]
        """

        self.image_size = tf.shape(inputs)[1:3]

        batch_norm_params = {
            'decay': self._BATCH_NORM_DECAY,
            'epsilon': 1e-05,
            'scale': True,
            'is_training': is_training,
            'fused': None,}

        #set activate function and parameters for conv2d and batch_normlization
        with slim.arg_scope([slim.conv2d, slim.batch_norm, common.fixed_padding], reuse=reuse):
            with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
                       normalizer_params=batch_norm_params,
                       biases_initializer=None,
                       activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=self._LEAKY_RELU)):
                with tf.variable_scope('darknet-53'):
                    route_1, route_2, route_3 = darknet53(inputs).outputs
                with tf.variable_scope('yolov3'):
                    route, inputs = self._yolo_block(inputs, 512)
                    feature_map_1 = self._detection_layer(inputs, self._ANCHORS[6:9])
                    feature_map_1 = tf.identity(feature_map_1, name='feature_map_1')

                    inputs = common.conv2d_fixed_padding(route, 256, 1)
                    upsample_size = route_2.get_shape().as_list()
                    inputs = self._upsample(inputs, upsample_size)
                    inputs = tf.concat([inputs, route_2], axis=3)

                    route, inputs = self._yolo_block(inputs, 256)
                    feature_map_2 = self._detection_layer(inputs, self._ANCHORS[3:6])
                    feature_map_2 = tf.identity(feature_map_2, name='feature_map_2')

                    inputs = common.conv2d_fixed_padding(route, 128, 1)
                    upsample_size = route_1.get_shape().as_list()
                    inputs = self._upsample(inputs, upsample_size)
                    inputs = tf.concat([inputs, route_1], axis=3)

                    route, inputs = self._yolo_block(inputs, 128)
                    feature_map_3 = self._detection_layer(inputs, self._ANCHORS[0:3])
                    feature_map_3 = tf.identity(feature_map_3, name='feature_map_3')

                return feature_map_1, feature_map_2, feature_map_3


    def _reorg_layer(self, feature_map, anchors):
        """
        regorg tensor
        :return:
        """
        num_anchors = len(anchors)
        grid_size = feature_map.shape.as_list()[1:3]
        stride = tf.cast(self.image_size // grid_size, tf.float32)
        feature_map = tf.reshape(feature_map, [-1, grid_size[0], grid_size[1], num_anchors, 5 + self._NUM_CLASSES])


        #boxes information
        box_centers, box_sizes , conf_logits, prob_logits = tf.split(
            feature_map, [2, 2, 1, self._NUM_CLASSES], axis=-1)
        box_centers = tf.nn.sigmoid(box_centers)
        grid_x = tf.range(grid_size[1], dtype=tf.float32)
        grid_y = tf.range(grid_size[0], dtype=tf.float32)

        a, b = tf.meshgrid(grid_x, grid_y)
        x_offset = tf.reshape(a, (-1, 1))
        y_offset = tf.reshape(b, (-1, 1))
        x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
        x_y_offset = tf.reshape(x_y_offset, [grid_size[0], grid_size[1], 1, 2])
        x_y_offset = tf.cast(x_y_offset, tf.float32)


        box_centers = box_centers + x_y_offset
        box_centers = box_centers * stride[::-1]
        box_sizes = tf.exp(box_sizes) * anchors  # anchors -> [w, h]
        boxes = tf.concat([box_centers, box_sizes], axis=-1)
        return x_y_offset, boxes, conf_logits, prob_logits


    def _reshape(self, x_y_offset, boxes, confs, probs):

        grid_size = x_y_offset.shape.as_list()[:2]
        boxes = tf.reshape(boxes, [-1, grid_size[0]*grid_size[1]*3, 4])
        confs = tf.reshape(confs, [-1, grid_size[0]*grid_size[1]*3, 1])
        probs = tf.reshape(probs, [-1, grid_size[0] * grid_size[1] * 3, self._NUM_CLASSES])

        return boxes, confs, probs


    def predict(self, feature_maps):
        """
        get boxes, confs and class probs from feature maps
        :param feature_maps:
        :return:
        """
        feature_map_1, feature_map_2, feature_map_3 = feature_maps
        feature_map_anchors = [(feature_map_1, self._ANCHORS[6:9]),
                               (feature_map_2, self._ANCHORS[3:6]),
                               (feature_map_3, self._ANCHORS[0:3])]
        results = [self._reorg_layer(feature_map, anchors) for (feature_map, anchors) in feature_map_anchors]
        boxes_list, confs_list, probs_list = [], [], []
        for result in results:
            boxes, conf_logits, prob_logits = self._reshape(*result)

            confs = tf.sigmoid(conf_logits)
            probs = tf.sigmoid(prob_logits)

            boxes_list.append(boxes)
            confs_list.append(confs)
            probs_list.append(probs)
        boxes = tf.concat(boxes_list, axis=1)
        confs = tf.concat(confs_list, axis=1)
        probs = tf.concat(probs_list, axis=1)

        center_x, center_y, width, height = tf.split(boxes, [1,1,1,1], axis=-1)
        x0 = center_x - width / 2.
        y0 = center_y - height / 2.
        x1 = center_x + width / 2.
        y1 = center_y + height / 2.

        boxes = tf.concat([x0, y0, x1, y1], axis=-1)
        return boxes, confs, probs


    def _broadcast_iou(self, true_box_xy, true_box_wh, pred_box_xy, pred_box_wh):
        """

        :param true_box_xy:
        :param true_box_wh: [V, 2]
        :param pred_box_xy:
        :param pred_box_wh: [N, 13, 13, 3, 2]
        :return: [N, 13, 13, 3, V]
        """
        #[N, 13, 13, 3, 2]
        pred_box_xy = tf.expand_dims(pred_box_xy, -2)
        pred_box_wh = tf.expand_dims(pred_box_wh, -2)
        #[1, V, 2]
        true_box_xy = tf.expand_dims(true_box_xy, 0)
        true_box_wh = tf.expand_dims(true_box_wh, 0)
        #[N, 13, 13, 3, 1, 2] & [1, V, 2] = [N, 13, 13, 3, V, 2]
        intersect_mins = tf.maximum(pred_box_xy - pred_box_wh / 2.,
                                    true_box_xy - true_box_wh / 2.)
        intersect_maxs = tf.minimum(pred_box_xy + pred_box_wh / 2.,
                                    true_box_xy + true_box_wh / 2.)
        intersect_wh = tf.maximum(intersect_maxs - intersect_mins, 0.)
        #[N, 13, 13, 3, V, 2]
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        #[N ,13 ,13 ,3 ,V]
        pre_box_area = pred_box_wh[..., 0] * pred_box_wh[..., 1]
        #[N, 13, 13, 3, 1]
        true_box_area = true_box_wh[..., 0] * true_box_wh[..., 1]
        #[1, V]

        iou = intersect_area / (pre_box_area + true_box_area - intersect_area)
        # [N, 13, 13, 3, 1] + [1, V] - [N, 13, 13, 3, V] / [N, 13, 13, 3, V]= [N, 13, 13, 3, V]
        return iou


    def loss_layer(self, feature_map_i, y_true, anchors):
        grid_size = tf.shape(feature_map_i)[1:3]
        grid_size_ = feature_map_i.shape.to_list()[1:3]

        y_true = tf.reshape(y_true, [-1, grid_size_[0], grid_size_[1], 3, 5+self._NUM_CLASSES])

        #the ratio in height and weight
        ratio = tf.cast(self.image_size / grid_size, tf.float32)
        #barch size
        N = tf.cast(tf.shape(feature_map_i)[0], tf.float32)

        x_y_offset, pred_boxes, pred_conf_logits, pred_prob_logits = self._reorg_layer(feature_map_i, anchors)

        object_mask = y_true[..., 4:5]
        # [N, 13, 13, 3]
        valid_true_boxes = tf.boolean_mask(y_true[..., 0:4], tf.cast(object_mask[..., 0], 'bool'))
        # [V, 4]
        valid_true_box_xy = valid_true_boxes[:, 0:2]
        valid_true_box_wh = valid_true_boxes[:, 2:4]
        pred_box_xy = pred_boxes[..., 0:2]
        pred_box_wh = pred_boxes[..., 2:4]
        # [N, 13, 13, 3, 2]
        iou = self._broadcast_iou(valid_true_box_xy, valid_true_box_wh, pred_box_xy, pred_box_wh)
        # [N, 13, 13, 3, V]
        best_iou = tf.reduce_max(iou, axis=-1)

        ignore_mask = tf.cast(best_iou < 0.5, tf.float32)
        ignore_mask = tf.expand_dims(ignore_mask, -1)
        # [N, 13, 13, 13, 1]
        true_xy = y_true[..., 0:2] / ratio[::-1] - x_y_offset
        pred_xy = pred_box_xy / ratio[::-1] - x_y_offset
        true_tw_th = y_true[..., 2:4] / anchors
        pred_tw_th = pred_box_wh / anchors
        # [N, 13, 13, 13, 2]
        true_tw_th = tf.where(condition=tf.equal(true_tw_th, 0),
                              x=tf.ones_like(true_tw_th), y=true_tw_th)
        pred_tw_th = tf.where(condition=tf.equal(pred_tw_th, 0),
                              x=tf.ones_like(pred_tw_th), y=pred_tw_th)
        true_tw_th = tf.log(tf.clip_by_value(true_tw_th, 1e-9, 1e9))
        pred_tw_th = tf.log(tf.clip_by_value(pred_tw_th, 1e-9, 1e9))

        box_loss_scale = 2. - (y_true[..., 2:3] / tf.cast(self.image_size[1]) * (y_true[..., 3:4] / tf.cast(self.image_size[0], tf.float32)))
        # [N, 13, 13, 3, 1]

        xy_loss = tf.reduce_sum(tf.square(true_xy - pred_xy) * object_mask * box_loss_scale) / N
        wh_loss = tf.reduce_sum(tf.square(true_tw_th - pred_tw_th) * object_mask * box_loss_scale) / N

        conf_pos_mask = object_mask
        conf_neg_mask = (1 - object_mask) * ignore_mask
        conf_loss_pos = conf_pos_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask,
                                                                                logits=pred_conf_logits)
        conf_loss_neg = conf_neg_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask,
                                                                                logits=pred_conf_logits)
        conf_loss = tf.reduce_sum(conf_loss_pos + conf_loss_neg) / N

        # shape: [N, 13, 13, 3, 1]
        class_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true[..., 5:],
                                                                           logits=pred_prob_logits)
        class_loss = tf.reduce_sum(class_loss) / N

        return xy_loss, wh_loss, conf_loss, class_loss


    def compute_loss(self, pred_feature_map, y_ture, ignore_thresh=0.5, max_box_per_image=8):
        """

        :param pred_feature_map:
        :param y_ture:
        :param ignore_thresh:
        :param max_box_per_image:
        :return:
        """
        loss_xy, loss_wh, loss_conf, loss_class = 0., 0., 0., 0.
        total_loss = 0.

        _ANCHORS = [self._ANCHORS[6:9], self._ANCHORS[3:6], self._ANCHORS[0:3]]

        for i in range(len(pred_feature_map)):
            result = self.loss_layer(pred_feature_map[i], y_ture[i], _ANCHORS[i])
            loss_xy += result[0]
            loss_wh += result[1]
            loss_conf += result[2]
            loss_class += result[3]
        total_loss = loss_xy + loss_wh + loss_conf + loss_class
        return [total_loss, loss_xy, loss_wh, loss_conf, loss_class]

