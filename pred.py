import os,sys,shutil
import tensorflow as tf
import numpy as np
import argparse
import cv2,math,glob,random,time
import time
import matplotlib.pyplot as plt
from model import *
from btgen import BatchGenerator

def tileImage(imgs):
    assert len(imgs.shape)==4
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

def loadModel(sess,saver,ckpt):
    if ckpt: # checkpointがある場合
        last_model = ckpt.model_checkpoint_path # 最後に保存したmodelへのパス
        #last_model = ckpt.all_model_checkpoint_paths[3]
        print ("load " + last_model)
        saver.restore(sess, last_model) # 変数データの読み込み
        print("succeed restore model")
    else:
        print("models were not found")
        init = tf.global_variables_initializer()
        sess.run(init)

def main(args):
    start = time.time()

    mode = args.mode
    res = args.res
    seed = args.seed
    tt_cutoff = args.cutoff
    tt_psi = args.psi
    noise_cutoff = args.noise
    SVIM_DIR = args.savedir
    SAVE_DIR = "model"
    stage = int(np.log2(res) - 1)

    if not os.path.exists(SVIM_DIR):
        os.mkdir(SVIM_DIR)

    if seed is not None:
        np.random.seed(seed)
        tf.set_random_seed(seed)
    else:
        seed = np.random.randint(0, 2147483648)
        np.random.seed(seed)
        tf.set_random_seed(seed)

    if mode == "uc":
        z = tf.placeholder(tf.float32, [None, 512])
        bcast= stage * 2
        alpha = tf.placeholder(tf.float32, [])
        #fakes = buildGenerator(z, alpha, stage=stage,isTraining=False)
        with tf.variable_scope("Generator") as scope:
            g_mapping = mapping(z,bcast)
            with tf.variable_scope("w_avg", reuse=tf.AUTO_REUSE):
                w_avg = tf.get_variable('w_avg', shape=[512], initializer=tf.initializers.zeros(), trainable=False)
            tt = truncation_trick(g_mapping, w_avg, bcast, psi=tt_psi, cutoff=tt_cutoff)
            g_synthesis = synthesis(tt, stage, alpha, use_noise=range(noise_cutoff,18))
            g_synthesis = tf.nn.tanh(g_synthesis)

        sess =tf.Session()
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(SAVE_DIR)
        loadModel(sess,saver,ckpt=ckpt)

        print("%.4e sec took initializing"%(time.time()-start))
        start = time.time()
        for i in range(0,10):
            z_batch = np.random.normal(0, 1,size=[25, 512])
            #g_image = fakes.eval(feed_dict={z: z_batch, alpha: 1}, session=sess)
            g_image = sess.run(g_synthesis, feed_dict={z:z_batch,alpha:1.0})
            cv2.imwrite(os.path.join(SVIM_DIR,"img_%d-%d.png"%(seed,i)),tileImage(g_image+1)*127.5)

    if mode == "sm":
        z = tf.placeholder(tf.float32, [None, 512])
        alpha = tf.placeholder(tf.float32, [])
        bcast= stage * 2
        style_ranges = [range(0, 4)] * 3 + [range(4, 8)] * 2 + [range(8, bcast)]
        w = tf.placeholder(tf.float32, [None, bcast, 512])
        src_latent = np.random.normal(0, 0.5, size=[6, 512])
        dst_latent = np.random.normal(0, 0.5, size=[6, 512])
        with tf.variable_scope("Generator") as scope:
            g_mapping = mapping(z,bcast)
            with tf.variable_scope("w_avg", reuse=tf.AUTO_REUSE):
                w_avg = tf.get_variable('w_avg', shape=[512], initializer=tf.initializers.zeros(), trainable=False)
            tt = truncation_trick(g_mapping, w_avg, bcast, psi=tt_psi, cutoff=tt_cutoff)
            g_synthesis = synthesis(w, stage, alpha, use_noise=range(noise_cutoff,18))
            g_synthesis = tf.nn.tanh(g_synthesis)

        sess =tf.Session()
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(SAVE_DIR)
        loadModel(sess,saver,ckpt)

        src_dlatent = sess.run(tt,feed_dict={z:src_latent})
        dst_dlatent = sess.run(tt,feed_dict={z:dst_latent})

        src_images = sess.run(g_synthesis, feed_dict={w:src_dlatent,alpha:1.0})
        dst_images = sess.run(g_synthesis, feed_dict={w:dst_dlatent,alpha:1.0})

        imgs = np.ones([res*7,res*7,3])
        for i in range(1,7):
            imgs[:res,i*res:(i+1)*res,:] = src_images[i-1]

        for i in range(1,7):
            row_dlatents = np.stack([dst_dlatent[i-1]] * 6)
            row_dlatents[:, style_ranges[i-1]] = src_dlatent[:, style_ranges[i-1]]

            g_images = sess.run(g_synthesis, feed_dict={w:row_dlatents,alpha:1.0})

            imgs[res*i:res*(i+1),:res,:] = dst_images[i-1]
            for j in range(1,7):
                imgs[res*i:res*(i+1),j*res:(j+1)*res,:] = g_images[j-1]

        cv2.imwrite(os.path.join(SVIM_DIR,"img_style_mixing_%d.png"%(seed)),(imgs+1)*127.5)
        print("saved as img_style_mixing_%d.png"%(seed))

    if mode == "tt":
        z = tf.placeholder(tf.float32, [None, 512])
        alpha = tf.placeholder(tf.float32, [])
        bcast= stage * 2
        w = tf.placeholder(tf.float32, [None, bcast, 512])
        psis = [-1, -0.67, -0.33, 0, 0.33, 0.67, 1]
        latent = np.random.normal(0, 0.5, size=[7, 512])
        with tf.variable_scope("Generator") as scope:
            g_mapping = mapping(z,bcast)
            with tf.variable_scope("w_avg", reuse=tf.AUTO_REUSE):
                w_avg = tf.get_variable('w_avg', shape=[512], initializer=tf.initializers.zeros(), trainable=False)
            g_synthesis = synthesis(w, stage, alpha, use_noise=range(noise_cutoff,18))
            g_synthesis = tf.nn.tanh(g_synthesis)

        sess =tf.Session()
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(SAVE_DIR)
        loadModel(sess,saver,ckpt)

        dlatent = sess.run(g_mapping,feed_dict={z:latent})
        w_avgnp = sess.run(w_avg)
        print(dlatent.shape)

        mlatent = np.zeros([49,stage*2,512])
        for i in range(7):
            for j,p in enumerate(psis):
                mlatent[i*7+j,:,:] = ((1-p)* w_avgnp + p* dlatent[i]).reshape(1,stage*2,512)

        mlatent = np.array(mlatent)
        print(mlatent.shape)

        g_image = sess.run(g_synthesis, feed_dict={w:mlatent,alpha:1.0})
        cv2.imwrite(os.path.join(SVIM_DIR,"img_truncation_trick_%d.png"%(seed)),tileImage(g_image+1)*127.5)
        print("saved as img_truncation_trick_%d.png"%(seed))

    print("%.4f sec took predicting"%(time.time()-start))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode","-m",dest="mode",type=str,default="uc",help="[uc->uncurated, sm->style_mix, tt->truncation_trick]")
    parser.add_argument("--res","-r",dest="res",type=int,default=256,help="[4,8,16...256,512,1024]")
    parser.add_argument('--seed',"-s",dest="seed", type=int, default=None, help='seed in the draw phase')
    parser.add_argument('--noise',"-n",dest="noise", type=int, default=4, help='noise cutoff in draw phase')
    parser.add_argument('--cutoff',"-c",dest="cutoff", type=int, default=8, help='truncation_trick cutoff in draw phase')
    parser.add_argument('--psi',"-p",dest="psi", type=float, default=0.7, help='truncation_trick psi in draw phase')
    parser.add_argument('--savedir',"-d",dest="savedir", type=str, default="generated", help="where images are saved")

    args = parser.parse_args()
    main(args=args)
