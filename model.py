import tensorflow as tf
import numpy as np

"""
common
"""
def _EQfc_variable(weight_shape, gain=2, name="fc"):
        with tf.variable_scope(name):
            # check weight_shape
            input_channels  = int(weight_shape[0])
            output_channels = int(weight_shape[1])
            c = np.sqrt(gain / input_channels)
            weight_shape    = (input_channels, output_channels)
            # define variables
            weight = tf.get_variable("w", weight_shape     ,
                                    initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0)) * c
            bias   = tf.get_variable("b", [weight_shape[1]],
                                    initializer=tf.constant_initializer(0.0))
        return weight, bias

def _EQconv_variable(weight_shape, gain=2, name="conv"):
    with tf.variable_scope(name):
        # check weight_shape
        w = int(weight_shape[0])
        h = int(weight_shape[1])
        input_channels  = int(weight_shape[2])
        output_channels = int(weight_shape[3])
        c = np.sqrt(gain / (input_channels * w * h))
        weight_shape = (w,h,input_channels, output_channels)
        # define variables
        weight = tf.get_variable("w", weight_shape     ,
                                initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0)) * c
        bias   = tf.get_variable("b", [output_channels],
                                initializer=tf.constant_initializer(0.0))
    return weight, bias

def _conv2d(x, W, stride, padding="SAME"):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = padding)

def _EQconv(x, input_layer, output_layer, stride=1, filter_size=3, name="conv" ,padding ="SAME", gain=2.0):
    conv_w, conv_b = _EQconv_variable([filter_size,filter_size,input_layer,output_layer], gain=gain, name=name)
    h = _conv2d(x ,conv_w, stride=stride, padding=padding) + conv_b
    return h

def _EQfc(x, input_layer, output_layer, name="fc"):
    fc_w, fc_b = _EQfc_variable([input_layer,output_layer],name=name)
    h = tf.matmul(x, fc_w) + fc_b
    return h

"""
function for generator
"""
def pixel_norm(x, epsilon=1e-8):
    return x / tf.sqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + epsilon)

def _up_sampling(x, ratio=2):
    x_b, x_h, x_w, x_c = x.get_shape().as_list()
    return tf.image.resize_bilinear(x, [x_h*ratio, x_w*ratio])

def add_noise(x, name, use_noise=True):
    noise = tf.random_normal([tf.shape(x)[0], x.shape[1], x.shape[2], 1], mean=0.0, stddev=0.5,)
    weight = tf.get_variable(name, shape=[x.get_shape().as_list()[-1]], initializer=tf.initializers.zeros())
    #weight = tf.get_variable(name, shape=[x.get_shape().as_list()[-1]], initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
    weight = tf.reshape(weight, [1, 1, 1, -1])
    if use_noise:
        print("%s uses noise"%name)
        return x + noise * weight
    else:
        print("%s doesn't use noise"%name)
        return x

def adain(x, w, name, epsilon=1e-8):
    # instance norm
    x -= tf.reduce_mean(x, axis=[1,2], keepdims=True)
    x *= tf.rsqrt(tf.reduce_mean(tf.square(x), axis=[1,2], keepdims=True) + epsilon)
    # style mod
    style = _EQfc(w, w.shape[1], x.shape[3]*2, name=name)
    style = tf.reshape(style, [-1, 1, 1, x.shape[3], 2])
    h = x * (style[:, :, :, :, 0] + 1) + style[:, :, :, :, 1]
    return h

def mapping(z, z_bCast):
    with tf.variable_scope("mapping", reuse=tf.AUTO_REUSE):
        h = pixel_norm(z)
        for i in range(8):
            h = _EQfc(h, 512, 512, "fc%d"%i)
            h = tf.nn.leaky_relu(h)
        h = tf.tile(h[:, np.newaxis], [1, z_bCast, 1])
    return h

def moving_average_of_w(w, w_avg, alpha=0.995):
    batch_avg = tf.reduce_mean(w[:, 0], axis=0)
    update_op = tf.assign(w_avg, batch_avg * (1 - alpha) + w_avg * alpha)
    with tf.control_dependencies([update_op]):
            w = tf.identity(w)
    return w

def style_mixing_regularization(w, z, z_bCast, style_mixing_prob=0.9):
    z2 = tf.random_normal(tf.shape(z), mean=0.0, stddev=0.5, dtype=tf.float32)
    w2 = mapping(z2, z_bCast)
    num_layers = z_bCast
    layer_idx = np.arange(num_layers)[np.newaxis, :, np.newaxis]
    mixing_cutoff = tf.cond(
            tf.random_uniform([], 0.0, 1.0) < style_mixing_prob,
            lambda: tf.random_uniform([], 1, num_layers, dtype=tf.int32),
            lambda: num_layers)
    w = tf.where(tf.broadcast_to(layer_idx < mixing_cutoff, tf.shape(w)), w, w2)
    return w

def truncation_trick(w, w_avg, z_bCast, psi=0.7, cutoff=8):
    num_layers = z_bCast
    layer_idx = np.arange(num_layers)[np.newaxis, :, np.newaxis]
    ones = np.ones(layer_idx.shape, dtype=np.float32)
    coefs = tf.where(layer_idx < cutoff, psi * ones, ones)
    new_w = w_avg * (1 - coefs) + w * coefs
    return new_w

def synthesis(w, stage, alpha, fn=[512, 512, 512, 512, 512, 256, 128, 64, 32, 16], use_noise=range(0,18)):
    rgb = 0
    for i in range(1, stage+1):
        #_reuse = False if stage == i else True
        if i==1:
            with tf.variable_scope("stage1", reuse=tf.AUTO_REUSE):
                h = tf.get_variable('constant', shape=[1, 4, 4, 512], initializer=tf.initializers.ones())
                #h = tf.get_variable('random', shape=[1, 4, 4, 512], initializer=tf.random_normal_initializer(mean=1.0, stddev=0.02))
                h = tf.tile(h, [tf.shape(w)[0], 1, 1, 1])
                h = add_noise(h, "noise_%d-1"%i, use_noise=(0 in use_noise))
                h = tf.nn.leaky_relu(h)
                h = adain(h, w[:,0], "adain_%d-1"%i)

                h = _EQconv(h, fn[i], fn[i], 1, 3, "conv2")
                h = add_noise(h, "noise_%d-2"%i, use_noise=(1 in use_noise))
                h = tf.nn.leaky_relu(h)
                h = adain(h, w[:,1], "adain_%d-2"%i)

                if stage == 1:
                    rgb = _EQconv(h, fn[i], 3, 1, 1, "toRGB", gain=1.0)
        else:
            with tf.variable_scope("stage%d"%i, reuse=tf.AUTO_REUSE):
                h = _up_sampling(h)

                if stage == i:
                    shortcut =  _EQconv(h, fn[i-1], 3, 1, 1, "shortcut", gain=1.0)

                h = _EQconv(h, fn[i-1], fn[i], 1, 3, "conv1")
                h = add_noise(h, "noise_%d-1"%i, use_noise=((i*2-2 in use_noise)))
                h = tf.nn.leaky_relu(h)
                h = adain(h, w[:,i*2-2], "adain_%d-1"%i)

                h = _EQconv(h, fn[i], fn[i], 1, 3, "conv2")
                h = add_noise(h, "noise_%d-2"%i, use_noise=((i*2-1 in use_noise)))
                h = tf.nn.leaky_relu(h)
                h = adain(h, w[:,i*2-1], "adain_%d-2"%i)

                if stage == i:
                    rgb = _EQconv(h, fn[i], 3, 1, 1, "toRGB", gain=1.0)
                    rgb = rgb * alpha + shortcut * (1 - alpha)
    return rgb
"""
function for discriminator
"""
def _avg_pool(x, k=2):
    return tf.nn.avg_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def minibatch_std(x):
    x_b, x_h, x_w, x_c = x.get_shape().as_list()
    group_size = tf.minimum(4, tf.shape(x)[0])
    y = tf.reshape(x, [group_size, -1, x_h, x_w, x_c]) #[GMHWC]
    mean = tf.reduce_mean(y, axis=0, keepdims=True)    #[GMHWC]
    y = y - mean                                       #[GMHWC]
    var = tf.reduce_mean(tf.square(y), axis=0)         #[MHWC]
    std = tf.sqrt(var + 1e-8)                          #[MHWC]
    ave = tf.reduce_mean(std, axis=[1,2,3], keepdims=True) # [M111]
    fmap = tf.tile(ave, [group_size, x_h, x_w, 1]) # [NHW1]
    return tf.concat([x, fmap], axis=3)

#####

def buildGenerator(z, alpha, stage, isTraining=True, psi=0.7, cutoff=8):
    fn = [512, 512, 512, 512, 512, 256, 128, 64, 32, 16]
    with tf.variable_scope("Generator") as scope:
        # mapping
        z_bCast = stage*2
        w = mapping(z, z_bCast)
        with tf.variable_scope("w_avg", reuse=tf.AUTO_REUSE):
            w_avg = tf.get_variable('w_avg', shape=[512], initializer=tf.initializers.zeros(), trainable=False)

        if isTraining:
            # moving average of w
            w = moving_average_of_w(w, w_avg)

            # style mixing regularization
            w = style_mixing_regularization(w, z, z_bCast)

        else:
            w = truncation_trick(w, w_avg, z_bCast, psi=psi, cutoff=cutoff)

        # synthesis
        rgb = synthesis(w, stage, alpha, fn)

    y = tf.nn.tanh(rgb)
    return y

def buildDiscriminator(y, alpha, stage, reuse):
    fn = [512, 512, 512, 512, 512, 256, 128, 64, 32, 16]
    with tf.variable_scope("Discriminator") as scope:
        for i in range(stage, 1, -1):
            _reuse = reuse if stage == i else True
            with tf.variable_scope("stage%d"%i, reuse=_reuse):
                if i == stage:
                    shortcut = _avg_pool(y)
                    shortcut = _EQconv(shortcut, 3, fn[i-1], 1, 3, "shortcut")
                    shortcut = tf.nn.leaky_relu(shortcut)

                    h = _EQconv(y, 3, fn[i], 1, 3, "fromRGB")
                    h = tf.nn.leaky_relu(h)

                    h = _EQconv(h, fn[i], fn[i], 1, 3, "conv1")
                    h = tf.nn.leaky_relu(h)

                    h = _EQconv(h, fn[i], fn[i-1], 1, 3, "conv2")
                    h = tf.nn.leaky_relu(h)

                    h = _avg_pool(h)
                    h = h * alpha + shortcut * (1 - alpha)
                else:
                    h = _EQconv(h, fn[i], fn[i], 1, 3, "conv1")
                    h = tf.nn.leaky_relu(h)

                    h = _EQconv(h, fn[i], fn[i-1], 1, 3, "conv2")
                    h = tf.nn.leaky_relu(h)

                    h = _avg_pool(h)

        _reuse = reuse if stage == 1 else True
        with tf.variable_scope("stage1", reuse=_reuse):
            if stage == 1:
                h = _EQconv(y, 3, fn[1], 1, 3, "fromRGB")
                h = tf.nn.leaky_relu(h)
            h = minibatch_std(h)

            h = _EQconv(h, fn[1]+1, fn[1], 1, 3, "conv1")
            h = tf.nn.leaky_relu(h)

            h = _EQconv(h, fn[1], fn[0], 1, 3, "conv2")
            h = tf.nn.leaky_relu(h)

            n_b, n_h, n_w, n_f = h.get_shape().as_list()
            h = tf.reshape(h,[-1,n_h*n_w*n_f])
            h = _EQfc(h, n_h*n_w*n_f, 1, "fc1")

    return h
