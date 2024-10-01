from __future__ import print_function
import os
import time
import random
import datetime
import scipy.misc
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from datetime import datetime
from util.util import *
from util.BasicConvLSTMCell import *


def conv_layer(inputs, filters, kernel_size, scope, stride=1, activation=tf.nn.relu, batch_norm=True, dropout=False, dropout_rate=0.5):
    with tf.variable_scope(scope):
        conv = slim.conv2d(inputs, filters, kernel_size, stride=stride, activation_fn=None, padding='SAME',
                           weights_initializer=tf.contrib.layers.xavier_initializer())
        if batch_norm:
            conv = slim.batch_norm(conv, activation_fn=activation)
        else:
            conv = activation(conv)
        if dropout:
            conv = slim.dropout(conv, keep_prob=1 - dropout_rate)
        return conv


def deconv_layer(inputs, filters, kernel_size, scope, stride=2, activation=tf.nn.relu, batch_norm=True, dropout=False, dropout_rate=0.5):
    with tf.variable_scope(scope):
        deconv = slim.conv2d_transpose(inputs, filters, kernel_size, stride=stride, activation_fn=None, padding='SAME',
                                       weights_initializer=tf.contrib.layers.xavier_initializer())
        if batch_norm:
            deconv = slim.batch_norm(deconv, activation_fn=activation)
        else:
            deconv = activation(deconv)
        if dropout:
            deconv = slim.dropout(deconv, keep_prob=1 - dropout_rate)
        return deconv


class DEBLUR(object):
    def __init__(self, args):
        self.args = args
        self.n_levels = 3
        self.scale = 0.5
        self.chns = 3 if self.args.model == 'color' else 1  # input / output channels

        # if args.phase == 'train':
        self.crop_size = 256
        self.data_list = open(args.datalist, 'rt').read().splitlines()
        self.data_list = list(map(lambda x: x.split(' '), self.data_list))
        random.shuffle(self.data_list)
        self.train_dir = os.path.join('./checkpoints', args.model)
        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)

        self.batch_size = args.batch_size
        self.epoch = args.epoch
        self.data_size = (len(self.data_list)) // self.batch_size
        self.max_steps = int(self.epoch * self.data_size)
        self.learning_rate = args.learning_rate

    def input_producer(self, batch_size=10):
        def read_data():
            img_a = tf.image.decode_image(tf.read_file(tf.string_join(['./training_set/', self.data_queue[0]])),
                                          channels=3)
            img_b = tf.image.decode_image(tf.read_file(tf.string_join(['./training_set/', self.data_queue[1]])),
                                          channels=3)
            img_a, img_b = preprocessing([img_a, img_b])
            return img_a, img_b

        def preprocessing(imgs):
            imgs = [tf.cast(img, tf.float32) / 255.0 for img in imgs]
            if self.args.model != 'color':
                imgs = [tf.image.rgb_to_grayscale(img) for img in imgs]
            img_crop = tf.unstack(tf.random_crop(tf.stack(imgs, axis=0), [2, self.crop_size, self.crop_size, self.chns]),
                                  axis=0)
            return img_crop

        with tf.variable_scope('input'):
            List_all = tf.convert_to_tensor(self.data_list, dtype=tf.string)
            gt_list = List_all[:, 0]
            in_list = List_all[:, 1]

            self.data_queue = tf.train.slice_input_producer([in_list, gt_list], capacity=20)
            image_in, image_gt = read_data()
            batch_in, batch_gt = tf.train.batch([image_in, image_gt], batch_size=batch_size, num_threads=8, capacity=20)

        return batch_in, batch_gt

    def resnet_block(self, inputs, filters, kernel_size, scope):
        with tf.variable_scope(scope):
            conv1 = slim.conv2d(inputs, filters, kernel_size, activation_fn=None, padding='SAME',
                                weights_initializer=tf.contrib.layers.xavier_initializer())
            conv1 = slim.batch_norm(conv1, activation_fn=tf.nn.relu)
            conv2 = slim.conv2d(conv1, filters, kernel_size, activation_fn=None, padding='SAME',
                                weights_initializer=tf.contrib.layers.xavier_initializer())
            conv2 = slim.batch_norm(conv2, activation_fn=None)
            return tf.nn.relu(inputs + conv2)

    def generator(self, inputs, reuse=False, scope='g_net'):
        n, h, w, c = inputs.get_shape().as_list()

        if self.args.model == 'lstm':
            with tf.variable_scope('LSTM'):
                cell = BasicConvLSTMCell([h // 4, w // 4], [3, 3], 128)
                rnn_state = cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)

        x_unwrap = []
        with tf.variable_scope(scope, reuse=reuse):
            with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                                activation_fn=None, padding='SAME', normalizer_fn=None,
                                weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                                biases_initializer=tf.constant_initializer(0.0)):

                inp_pred = inputs
                for i in range(self.n_levels):
                    scale = self.scale ** (self.n_levels - i - 1)
                    hi = int(round(h * scale))
                    wi = int(round(w * scale))
                    inp_blur = tf.image.resize_images(inputs, [hi, wi], method=tf.image.ResizeMethod.BILINEAR)
                    inp_pred = tf.stop_gradient(tf.image.resize_images(inp_pred, [hi, wi], method=tf.image.ResizeMethod.BILINEAR))
                    inp_all = tf.concat([inp_blur, inp_pred], axis=3, name='inp')
                    if self.args.model == 'lstm':
                        rnn_state = tf.image.resize_images(rnn_state, [hi // 4, wi // 4], method=tf.image.ResizeMethod.BILINEAR)

                    # Encoder
                    conv = conv_layer(inp_all, 64, [3, 3], scope='enc_initial', batch_norm=True)
                    for j in range(6):  # Increased number of ResNet blocks
                        conv = self.resnet_block(conv, 64, 3, scope='enc_resblock_%d' % j)
                    conv = conv_layer(conv, 128, [3, 3], scope='enc_downsample_1', stride=2)
                    for j in range(6):
                        conv = self.resnet_block(conv, 128, 3, scope='enc_resblock_128_%d' % j)
                    conv = conv_layer(conv, 256, [3, 3], scope='enc_downsample_2', stride=2)
                    for j in range(6):
                        conv = self.resnet_block(conv, 256, 3, scope='enc_resblock_256_%d' % j)

                    if self.args.model == 'lstm':
                        deconv, rnn_state = cell(conv, rnn_state)
                    else:
                        deconv = conv

                    # Decoder
                    for j in range(6):
                        deconv = self.resnet_block(deconv, 256, 3, scope='dec_resblock_256_%d' % j)
                    deconv = deconv_layer(deconv, 128, [3, 3], scope='dec_up_1')
                    deconv += conv_layer(conv_layer(deconv, 128, [3, 3], scope='dec_skip_1'), 128, [3, 3], scope='dec_skip_1_bn')
                    for j in range(6):
                        deconv = self.resnet_block(deconv, 128, 3, scope='dec_resblock_128_%d' % j)
                    deconv = deconv_layer(deconv, 64, [3, 3], scope='dec_up_2')
                    deconv += conv_layer(conv_layer(deconv, 64, [3, 3], scope='dec_skip_2'), 64, [3, 3], scope='dec_skip_2_bn')
                    for j in range(6):
                        deconv = self.resnet_block(deconv, 64, 3, scope='dec_resblock_64_%d' % j)
                    deconv = conv_layer(deconv, self.chns, [3, 3], scope='dec_final', activation=tf.identity, batch_norm=False)

                    inp_pred = deconv  # Update prediction

                    if i >= 0:
                        x_unwrap.append(inp_pred)
                    if i == 0:
                        tf.get_variable_scope().reuse_variables()

        return x_unwrap

    def build_model(self):
        img_in, img_gt = self.input_producer(self.batch_size)

        tf.summary.image('img_in', im2uint8(img_in))
        tf.summary.image('img_gt', im2uint8(img_gt))
        print('img_in, img_gt', img_in.get_shape(), img_gt.get_shape())

        # generator
        x_unwrap = self.generator(img_in, reuse=False, scope='g_net')
        # calculate multi-scale loss
        self.loss_total = 0
        for i in range(self.n_levels):
            _, hi, wi, _ = x_unwrap[i].get_shape().as_list()
            gt_i = tf.image.resize_images(img_gt, [hi, wi], method=tf.image.ResizeMethod.BILINEAR)
            loss = tf.reduce_mean(tf.square(gt_i - x_unwrap[i]))
            self.loss_total += loss

            tf.summary.image('out_' + str(i), im2uint8(x_unwrap[i]))
            tf.summary.scalar('loss_' + str(i), loss)

        # losses
        tf.summary.scalar('loss_total', self.loss_total)

        # training vars
        all_vars = tf.trainable_variables()
        self.all_vars = all_vars
        self.g_vars = [var for var in all_vars if 'g_net' in var.name]
        self.lstm_vars = [var for var in all_vars if 'LSTM' in var.name]
        for var in all_vars:
            print(var.name)

    def train(self):
        def get_optimizer(loss, global_step=None, var_list=None, is_gradient_clip=False):
            optimizer = tf.train.AdamOptimizer(self.lr)
            if is_gradient_clip:
                grads_and_vars = optimizer.compute_gradients(loss, var_list=var_list)
                unchanged_gvs = [(grad, var) for grad, var in grads_and_vars if 'LSTM' not in var.name]
                rnn_grads = [grad for grad, var in grads_and_vars if 'LSTM' in var.name]
                rnn_vars = [var for grad, var in grads_and_vars if 'LSTM' in var.name]
                capped_grads, _ = tf.clip_by_global_norm(rnn_grads, clip_norm=3)
                capped_gvs = list(zip(capped_grads, rnn_vars))
                train_op = optimizer.apply_gradients(capped_gvs + unchanged_gvs, global_step=global_step)
            else:
                train_op = optimizer.minimize(loss, global_step, var_list)
            return train_op

        global_step = tf.Variable(initial_value=0, dtype=tf.int32, trainable=False)
        self.global_step = global_step

        # build model
        self.build_model()

        # learning rate decay
        self.lr = tf.train.polynomial_decay(self.learning_rate, global_step, self.max_steps, end_learning_rate=0.0,
                                            power=0.3)
        tf.summary.scalar('learning_rate', self.lr)

        # training operators
        train_gnet = get_optimizer(self.loss_total, global_step, self.all_vars, is_gradient_clip=True)

        # session and thread
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess = sess
        sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=50, keep_checkpoint_every_n_hours=1)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # training summary
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(self.train_dir, sess.graph, flush_secs=30)

        for step in range(sess.run(global_step), self.max_steps + 1):
            print("Epochs ")
            print(list(range(sess.run(global_step), self.max_steps + 1)))
            start_time = time.time()

            # update G network
            _, loss_total_val = sess.run([train_gnet, self.loss_total])

            duration = time.time() - start_time
            # print loss_value
            assert not np.isnan(loss_total_val), 'Model diverged with loss = NaN'

            if step % 5 == 0:
                num_examples_per_step = self.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = %.5f (%.1f data/s; %.3f s/bch)')
                print(format_str % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), step, loss_total_val,
                                    examples_per_sec, sec_per_batch))

            if step % 20 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, global_step=step)

            # Save the model checkpoint periodically.
            if step % 1000 == 0 or step == self.max_steps:
                checkpoint_path = os.path.join(self.train_dir, 'checkpoints')
                self.save(sess, checkpoint_path, step)

        coord.request_stop()
        coord.join(threads)

    def save(self, sess, checkpoint_dir, step):
        model_name = "deblur.model"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    def load(self, sess, checkpoint_dir, step=None):
        print(" [*] Reading checkpoints...")
        model_name = "deblur.model"
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

        if step is not None:
            ckpt_name = model_name + '-' + str(step)
            self.saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Reading intermediate checkpoints... Success")
            return str(step)
        elif ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            ckpt_iter = ckpt_name.split('-')[1]
            self.saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Reading updated checkpoints... Success")
            return ckpt_iter
        else:
            print(" [*] Reading checkpoints... ERROR")
            return False

    def test(self, height, width, input_path, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        imgsName = sorted(os.listdir(input_path))

        H, W = height, width
        inp_chns = 3 if self.args.model == 'color' else 1
        self.batch_size = 1
        inputs = tf.placeholder(shape=[self.batch_size, H, W, inp_chns], dtype=tf.float32)
        outputs = self.generator(inputs, reuse=True)

        sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

        self.saver = tf.train.Saver()
        self.load(sess, self.train_dir)
        print(imgsName)
        for imgName in imgsName:
            blur = scipy.misc.imread(os.path.join(input_path, imgName))
            h, w, c = blur.shape
            # make sure the width is larger than the height
            rot = False
            if h > w:
                blur = np.transpose(blur, [1, 0, 2])
                rot = True
            h = blur.shape[0]
            w = blur.shape[1]
            resize = False
            if h > H or w > W:
                scale = min(1.0 * H / h, 1.0 * W / w)
                new_h = int(h * scale)
                new_w = int(w * scale)
                blur = scipy.misc.imresize(blur, [new_h, new_w], 'bicubic')
                resize = True
                blurPad = np.pad(blur, ((0, H - new_h), (0, W - new_w), (0, 0)), 'edge')
            else:
                blurPad = np.pad(blur, ((0, H - h), (0, W - w), (0, 0)), 'edge')
            blurPad = np.expand_dims(blurPad, 0)
            if self.args.model != 'color':
                blurPad = np.transpose(blurPad, (0, 2, 3, 1))  # Corrected transpose

            start = time.time()
            deblur = sess.run(outputs, feed_dict={inputs: blurPad / 255.0})
            duration = time.time() - start
            print('Saving results: %s ... %4.3fs' % (os.path.join(output_path, imgName), duration))
            res = deblur[-1]
            if self.args.model != 'color':
                res = np.transpose(res, (0, 2, 3, 1))
            res = im2uint8(res[0, :, :, :])
            # crop the image into original size
            if resize:
                res = res[:new_h, :new_w, :]
                res = scipy.misc.imresize(res, [h, w], 'bicubic')
            else:
                res = res[:h, :w, :]

            if rot:
                res = np.transpose(res, [1, 0, 2])
            scipy.misc.imsave(os.path.join(output_path, imgName), res)
