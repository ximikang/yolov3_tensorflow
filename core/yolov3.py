import tensorflow as tf
from core import common

slim = tf.contrib.slim

class darnet53(object):
    """
    define darnet53 network 
    """
    
    def __init__(self, input):
        pass
    

    def _darknet_block(self, inputs, filters):
        shortcut = inputs

class yolov3(object):

    
    def __init__(self, num_classes, anchors, batch_norm_decay=0.9, leaky_relu=0.1):
        self.num_classes = num_classes
        self.anchors = anchors
        self.batch_norm_decay = batch_norm_decay
        self.leaky_relu = leaky_relu
        self.feature_maps = []
        #[none, 13, 13 255], [none, 26, 26, 255], [none, 52, 52, 255]
