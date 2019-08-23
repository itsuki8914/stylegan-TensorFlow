import os,sys,shutil
import tensorflow as tf
import numpy as np
import argparse
import cv2,math,glob,random,time
import time
import matplotlib.pyplot as plt
from model import *
from btgen import BatchGenerator

DATASET_DIR = "ffhq_dataset"
SAVE_DIR = "model"
SVIM_DIR = "sample"

def tileImage(imgs):
    d = int(math.sqrt(imgs.shape[0]-1))+1
    h = imgs[0].shape[0]
    w = imgs[0].shape[1]
    r = np.zeros((h*d,w*d,3),dtype=np.float32)
    for idx,img in enumerate(imgs):
        idx_y = int(idx/d)
        idx_x = idx-idx_y*d
        r[idx_y*h:(idx_y+1)*h,idx_x*w:(idx_x+1)*w,:] = img
    return r

def printParam(scope):
    total_parameters = 0
    for variable in tf.trainable_variables(scope=scope):
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print("{} has {} parameters".format(scope, total_parameters))

def calc_losses(d_reals, d_fakes, xhats, d_xhats):
    g_losses = []
    d_losses = []
    for d_real, d_fake, xhat, d_xhat in zip(d_reals, d_fakes, xhats, d_xhats):
        g_loss = -tf.reduce_mean(d_fake)
        d_loss = tf.reduce_mean(d_fake) - tf.reduce_mean(d_real)

        drift_loss = tf.reduce_mean(d_real ** 2 * 1e-3)
        d_loss += drift_loss

        scale = 10.0
        grad = tf.gradients(d_xhat, [xhat])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1, 2, 3]))
        gradient_penalty = tf.reduce_mean(tf.square(slopes - 1.0) * scale)
        d_loss += gradient_penalty

        g_losses.append(g_loss)
        d_losses.append(d_loss)

    return g_losses, d_losses

def main():
    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)

    if not os.path.exists(SVIM_DIR):
        os.mkdir(SVIM_DIR)

    img_size = [2**(i+2) for i in range(9)]
    #bs = [64, 48, 32, 24, 16, 12, 8, 4, 4] # PC has enough VRAM
    #bs = [48, 32, 24, 16, 12, 8, 4, 4, 4]
    bs = [16, 16, 16, 16, 12, 8, 4, 3, 2]
    #steps = [16000,24000,40000,64000,96000,128000,160000,200000,240000]
    steps = [1,16000,24000,40000,64000,96000,128000,192000,320000]
    #steps = [12000,28000,60000,120000,240000,360000,600000,960000,2160000]

    z_dim = 512

    # loading images on training
    batch = BatchGenerator(img_size=512,datadir=DATASET_DIR)
    IN_ = batch.getBatch(4)
    IN_ = (IN_ + 1)*127.5
    IN_ =tileImage(IN_)
    cv2.imwrite("{}/input.png".format(SVIM_DIR),IN_)

    z = tf.placeholder(tf.float32, [None, z_dim])
    X_real =  [tf.placeholder(tf.float32, [None, r, r, 3]) for r in img_size]
    alpha = tf.placeholder(tf.float32, [])
    X_fake = [buildGenerator(z, alpha, stage=i+1) for i in range(9)]
    fake_y = [buildDiscriminator(x, alpha, stage=i+1, reuse=False) for i, x in enumerate(X_fake)]
    real_y = [buildDiscriminator(x, alpha, stage=i+1, reuse=True) for i, x in enumerate(X_real)]
    lr = tf.placeholder(tf.float32, [])
    """
    #WGAN-gp
    xhats = []
    d_xhats = []
    for i, (real, fake) in enumerate(zip(X_real, X_fake)):
        epsilon = tf.random_uniform(shape=[tf.shape(real)[0], 1, 1, 1], minval=0.0, maxval=1.0)
        inter = real * epsilon + fake * (1 - epsilon)
        d_xhat = buildDiscriminator(inter, alpha, stage=i+1, reuse=True)
        xhats.append(inter)
        d_xhats.append(d_xhat)

    g_losses, d_losses = calc_losses(real_y, fake_y, xhats, d_xhats)
    """

    # softplus
    g_losses = []
    d_losses = []
    for i, (real_images, real_logit, fake_logit) in enumerate(zip(X_real, real_y, fake_y)):
        r1_gamma = 10.0

        # discriminator loss: gradient penalty
        d_loss_gan = tf.nn.softplus(fake_logit) + tf.nn.softplus(-real_logit)
        real_loss = tf.reduce_sum(real_logit)
        real_grads = tf.gradients(real_loss, [real_images])[0]
        r1_penalty = tf.reduce_sum(tf.square(real_grads), axis=[1, 2, 3])
        d_loss = d_loss_gan + r1_penalty * (r1_gamma * 0.5)
        d_loss = tf.reduce_mean(d_loss)

        # generator loss: logistic nonsaturating
        g_loss = tf.nn.softplus(-fake_logit)
        g_loss = tf.reduce_mean(g_loss)
        g_losses.append(g_loss)
        d_losses.append(d_loss)

    g_var = [x for x in tf.trainable_variables() if "Generator"     in x.name]
    d_var = [x for x in tf.trainable_variables() if "Discriminator" in x.name]
    opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.0, beta2=0.99, epsilon=1e-8)

    g_opt = [opt.minimize(g_loss, var_list=g_var) for g_loss in g_losses]
    d_opt = [opt.minimize(d_loss, var_list=d_var) for d_loss in d_losses]

    printParam(scope="Generator")
    printParam(scope="Discriminator")

    start = time.time()

    config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.75))

    sess =tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    ckpt = tf.train.get_checkpoint_state(SAVE_DIR)

    if ckpt: # checkpointがある場合
        last_model = ckpt.model_checkpoint_path # 最後に保存したmodelへのパス
        print ("load " + last_model)
        saver.restore(sess, last_model) # 変数データの読み込み
        print("succeed restore model")
    else:
        print("models were not found")
        init = tf.global_variables_initializer()
        sess.run(init)

    print("%.4f sec took initializing"%(time.time()-start))

    start = time.time()
    for stage in range(0,9):
        #batch =  BatchGenerator(img_size=img_size[stage],datadir=DATASET_DIR)
        if stage<6:
            batch =  BatchGenerator(img_size=img_size[stage],datadir="ffhq_dataset128")
        else:
            batch =  BatchGenerator(img_size=img_size[stage],datadir="ffhq_dataset")
        #save samples
        x_batch = batch.getBatch(bs[stage],alpha=1.0)
        out = tileImage(x_batch)
        out = np.array((out + 1) * 127.5, dtype=np.uint8)
        outdir = os.path.join(SVIM_DIR, 'stage{}'.format(stage+1))
        os.makedirs(outdir, exist_ok=True)
        dst = os.path.join(outdir, 'sample.png')
        cv2.imwrite(dst, out)

        trans_lr = 1e-3
        g_hist = []
        d_hist = []
        print("starting stage{}".format(stage+1))
        for i in range(steps[stage]+1):
            delta = 4*i/(steps[stage]+1)
            # First stage does not require interpolation
            if stage == 1 or stage == 2:
                alp = 1.0
            else:
                alp = min(delta, 1.0)

            x_batch = batch.getBatch(bs[stage],alpha=alp)

            z_batch = np.random.normal(0, 0.5, [bs[stage], z_dim])

            _, dis_loss = sess.run([d_opt[stage], d_losses[stage]],
                                 feed_dict={X_real[stage]: x_batch, z: z_batch, alpha: alp, lr:trans_lr})

            z_batch = np.random.normal(0, 0.5, [bs[stage], z_dim])
            _, gen_loss = sess.run([g_opt[stage], g_losses[stage]], feed_dict={z: z_batch, alpha: alp, lr:trans_lr})

            g_hist.append(gen_loss)
            d_hist.append(dis_loss)

            print("stage:[%d], in step %s, dis_loss = %.3e, gen_loss = %.3e, alpha = %.3f, lr = %.3e"
                    %(stage+1, i,dis_loss, gen_loss, alp, trans_lr))

            if alp==1.0:
                #decaying learning rate
                trans_lr *= (1 - 2 / steps[stage])

            if i%100 == 0:
                z_batch = np.random.normal(0, 0.5, [bs[stage], z_dim])
                out = X_fake[stage].eval(feed_dict={z: z_batch, alpha: alp}, session=sess)
                out = tileImage(out)
                out = np.array((out + 1) * 127.5, dtype=np.uint8)
                outdir = os.path.join(SVIM_DIR, 'stage{}'.format(stage+1))
                os.makedirs(outdir, exist_ok=True)
                dst = os.path.join(outdir, '{}_alp.png'.format('{0:09d}'.format(i)))
                cv2.imwrite(dst, out)

                fig = plt.figure(figsize=(8,6), dpi=128)
                ax = fig.add_subplot(111)
                plt.title("Loss")
                plt.grid(which="both")
                plt.yscale("log")
                ax.plot(g_hist,label="gen_loss", linewidth = 0.25)
                ax.plot(d_hist,label="dis_loss", linewidth = 0.25)
                plt.xlabel('step', fontsize = 16)
                plt.ylabel('loss', fontsize = 16)
                plt.legend(loc = 'upper right')
                plt.savefig(os.path.join(outdir,"hist.png"))
                plt.close()

            if i % 8000 == 0 and i!=0:
                saver.save(sess,os.path.join(SAVE_DIR,"model.ckpt"),i)


if __name__ == '__main__':
    main()
