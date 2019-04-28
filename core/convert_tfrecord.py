import sys
import numpy as np
import tensorflow as tf

dataset_txt = './raccoon_dataset/train.txt'
tfrecord_path_prefix = './raccoon_dataset/train'
def main(dataset_txt, tfrecord_path_prefix):
    dataset = {}
    with open(dataset_txt, 'r') as f:
        for line in f.readlines():
            example = line.split(' ')
            image_path = example[0]
            boxes_num = len(example[1:]) // 5
            boxes = np.zeros([boxes_num, 5], dtype=np.float32)
            for i in range(boxes_num):
                boxes[i] = example[1 + i * 5:6 + i * 5]
            dataset[image_path] = boxes
    image_paths = list(dataset.keys())
    image_num = len(image_paths)

    tfrecord_file = tfrecord_path_prefix + '.tfrecords'

    with tf.python_io.TFRecordWriter(tfrecord_file) as record_writer:
        for i in range(image_num):
            image = tf.gfile.FastGFile(image_paths[i], 'rb').read()
            boxes = dataset[image_paths[i]]
            boxes = boxes.tostring()
            example = tf.train.Example(features=tf.train.Features(
                feature={
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                    'boxes': tf.train.Feature(bytes_list=tf.train.BytesList(value=[boxes])),
                }
            ))

            record_writer.write(example.SerializeToString())
        print(">> Saving %d images in %s" %(image_num, tfrecord_file))


if __name__ == '__main__':
    main(dataset_txt, tfrecord_path_prefix)