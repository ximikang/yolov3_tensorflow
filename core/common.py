import tensorflow as tf
slim = tf.contrib.slim


def conv2d_fixed_padding(inputs, filters, kernel_size, strides=1):
    if strides > 1:
        inputs = fixed_padding(inputs,kernel_size)
    inputs = slim.conv2d(inputs, filters, kernel_size, stride=strides,
                         padding=('SAME' if strides==1 else 'VALID'))

@tf.contrib.framework.add_arg_scope
def fixed_padding(inputs, kernel_size, *args, mode="CONSTANT", **kwargs):

    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    padded_input = tf.pad(inputs, [[0,0],[pad_beg, pad_end],
                                   [pad_beg, pad_end], [0, 0]], mode=mode)
    return padded_input