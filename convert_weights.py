import os
import sys
import wget
import time
import tensorflow as tf
from core import yolov3, utils


image_h , image_w = 416, 416
num_classes = 80
iou_threshold = 0.5
score_threshold = 0.5
ckpt_file = './checkpoint/yolov3.ckpt'
anchors_path = './data/raccoon_anchors.txt'
weights_path = './checkpoint/yolov3.weights'




if __name__ == '__main__':
    print("the input image size is {}*{}".format(image_h, image_w))
    anchors = utils.get_anchors(anchors_path, image_h, image_w)
    model = yolov3.yolov3(num_classes, anchors)


    with tf.Graph().as_default() as graph:
        sess = tf.Session(graph=graph)
        inputs = tf.placeholder(tf.float32, [1, image_h, image_w, 3])
        print(inputs)

        with tf.variable_scope('yolov3'):
            feature_map = model.forward(inputs, is_training=False)
        boxes ,confs, probs = model.predict(feature_map)
        scores = confs * probs

        print("=>", boxes.name[:-2], scores.name[:-2])
        cpu_out_node_names = [boxes.name[:-2], scores.name[:-2]]
        boxes, scores, labels = utils.gpu_nms(boxes, scores, num_classes,
                                              score_thresh=score_threshold,
                                              iou_thresh=score_threshold)
        print("=>", boxes.name[:-2], scores.name[:-2], labels.name[:-2])
        gpu_out_node_names = [boxes.name[:-2], scores.name[:-2], labels.name[:-2]]
        feature_map_1, feature_map_2, feature_map_3 = feature_map
        saver = tf.train.Saver(var_list=tf.global_variables(scope='yolov3'))


        if not os.path.exists(weights_path):
            print('=> weight file not exist!!!')
        else:
            load_ops = utils.load_weights(tf.global_variables(scope='yolov3'), weights_path)
            sess.run(load_ops)
            save_path = saver.save(sess, save_path=ckpt_file)
            print("=> model saved in path: {}".format(save_path))



