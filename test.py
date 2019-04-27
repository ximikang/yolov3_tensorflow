import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


grid_size = [10, 15]

grid_x = tf.range(grid_size[1], dtype=tf.float32)
grid_y = tf.range(grid_size[0], dtype=tf.float32)

a, b = tf.meshgrid(grid_x, grid_y)
x_offset = tf.reshape(a, (-1, 1))
y_offset = tf.reshape(b, (-1, 1))

x_y_offset = tf.concat([x_offset, y_offset], axis=-1)

x_y_offset = tf.reshape(x_y_offset, [grid_size[0], grid_size[1], 1, 2])
x_y_offset = tf.cast(x_y_offset, tf.float32)

centers = tf.constant(10, shape=[10,10,15,1,5], dtype=tf.float32)
centers, dsj = tf.split(centers, [2,3], axis=-1)
print(centers.shape)

centers = tf.nn.sigmoid(centers)
centers = tf.cast(centers, tf.float32)
stride = [15,10]
centers = centers * stride[::-1]
#boxes = tf.exp([box_centers, box_sizes], axis=-1)

print(x_y_offset, centers)
print(x_y_offset + centers)
