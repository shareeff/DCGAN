import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import datetime
#import matplotlib.pyplot as plt
import scipy.misc
import numpy as np


class DCGAN():

    def __init__(self, args):
       self.z_dim = args.z_dim
       self.batch_size = args.batch_size
       self.data_dir = args.data_dir
       self.checkpoint_dir = args.checkpoint_dir
       self.result_dir = args.result_dir
       self.initializer = tf.truncated_normal_initializer(stddev=0.02)
       #self.noise_input = tf.truncated_normal([self.batch_size, self.z_dim], mean=0, stddev=1, name='z')

       self.z = tf.placeholder('float', [None, self.z_dim])
       self.x = tf.placeholder('float', [None, 28, 28, 1])
       self.generate_model()



    def lrelu(self, x, leak=0.2, name='lrelu'):
        with tf.variable_scope(name):
            f1 = 0.5 * (1 + leak)
            f2 = 0.5 * (1 - leak)
            return f1 * x + f2 * abs(x)

    def discriminator(self, x, reuse=False):
        with tf.variable_scope('discriminator'):
            d_conv1 = tf.contrib.layers.conv2d(inputs=x, num_outputs=28, kernel_size=3, stride=2, \
                                                    padding='SAME', reuse=reuse, activation_fn=self.lrelu, \
                                                    weights_initializer=self.initializer, scope='d_conv1')       # 14x14x28
            d_conv2 = tf.contrib.layers.conv2d(inputs=d_conv1, num_outputs=28*2, kernel_size=3, stride=2, \
                                                    padding='SAME', reuse=reuse, activation_fn=self.lrelu,
                                                    normalizer_fn = tf.contrib.layers.batch_norm, \
                                                    weights_initializer=self.initializer, scope='d_conv2')  # 7x7x28x2
            d_conv3 = tf.contrib.layers.conv2d(inputs=d_conv2, num_outputs=28*4, kernel_size=3, stride=2, \
                                                    padding='SAME', reuse=reuse, activation_fn=self.lrelu, \
                                                    normalizer_fn = tf.contrib.layers.batch_norm, \
                                                    weights_initializer = self.initializer, scope='d_conv3') # 4x4x28x4
            d_conv4 = tf.contrib.layers.conv2d(inputs=d_conv3, num_outputs=28*8, kernel_size=3, stride=1, \
                                                   padding='SAME', reuse=reuse, activation_fn=self.lrelu, \
                                                   normalizer_fn = tf.contrib.layers.batch_norm, \
                                                   weights_initializer = self.initializer, scope='d_conv4') # 4x4x28x8
            d_fc1 = tf.reshape(d_conv4, shape=[-1, 3584])
            d_fc1 = tf.contrib.layers.fully_connected(inputs=d_fc1, num_outputs=1, reuse=reuse, \
                                                           activation_fn=self.lrelu, \
                                                           weights_initializer=self.initializer, scope='d_fc1')
            return d_fc1

    def generator(self,z):
        with tf.variable_scope('generator'):
            g_w1 = tf.Variable(tf.truncated_normal([self.z_dim, 3584], stddev=0.02), name='g_w1')
            g_b1 = tf.Variable(tf.constant(0.0, shape=[3584]), name='b_w1')
            g1 = tf.matmul(z, g_w1) + g_b1
            g1 = tf.reshape(g1, [-1, 4, 4, 28*8])
            g_conv_trans1 = tf.contrib.layers.conv2d_transpose(inputs=g1, num_outputs=28*4, kernel_size=3, stride=1, \
                                                    padding='SAME', reuse=False, activation_fn= tf.nn.relu, \
                                                    normalizer_fn=tf.contrib.layers.batch_norm, \
                                                    weights_initializer=self.initializer, scope='g_conv_trans1')           # 4x4x28x4
            g_conv_trans1 = tf.reshape(g_conv_trans1, shape=[-1, 4, 4, 28*4])
            g_conv_trans2 = tf.contrib.layers.conv2d_transpose(inputs=g_conv_trans1, num_outputs=28*2, kernel_size=3, stride=2, \
                                                    padding='SAME', reuse=False, activation_fn=tf.nn.relu, \
                                                    normalizer_fn = tf.contrib.layers.batch_norm, \
                                                    weights_initializer=self.initializer, scope='g_conv_trans2')       #8x8x28x2
            g_conv_trans2 = tf.slice(g_conv_trans2, [0, 1, 1, 0], [-1, 7, 7, 28*2])
           # g_conv_trans2 = tf.reshape(g_conv_trans2, shape=[-1, 7, 7, 28*2])
            g_conv_trans3 = tf.contrib.layers.conv2d_transpose(inputs=g_conv_trans2, num_outputs=28*1, kernel_size=3, stride=2, \
                                                    padding='SAME', reuse=False, activation_fn= tf.nn.relu, \
                                                    normalizer_fn = tf.contrib.layers.batch_norm, \
                                                    weights_initializer=self.initializer, scope='g_conv_trans3')   #7x7x128
           # g_conv_trans3 = tf.reshape(g_conv_trans3, shape=[-1, 14, 14, 28*1])
            g_conv_trans4 = tf.contrib.layers.conv2d_transpose(inputs=g_conv_trans3, num_outputs=1, kernel_size=3, stride=2, \
                                                    padding='SAME', reuse=False, activation_fn = tf.nn.tanh, \
                                                    weights_initializer=self.initializer, scope='g_conv_trans4')
           # g_conv_trans4 = tf.reshape(g_conv_trans4, shape=[-1, 28, 28, 1])

            return g_conv_trans4


    def generate_model(self):
        self.Gz = self.generator(self.z)
        self.Dx = self.discriminator(self.x)
        self.Dg = self.discriminator(self.Gz, reuse=True)
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Dg, labels=tf.ones_like(self.Dg)))
        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Dx, \
                                                                                  labels=tf.fill([self.batch_size, 1], 0.9)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Dg, \
                                                                                  labels=tf.zeros_like(self.Dg)))
        self.d_loss = self.d_loss_real + self.d_loss_fake
        self.d_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        self.g_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        self.d_optimizer_real = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(self.d_loss_real, var_list=self.d_var)
        self.d_optimizer_fake = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(self.d_loss_fake, var_list=self.d_var)
        self.d_optimizer = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(self.d_loss, var_list=self.d_var)
        self.g_optimizer = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(self.g_loss, var_list=self.g_var)

    def train(self):
        self.load_mnist()
        self.saver = tf.train.Saver()
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        #z = tf.truncated_normal([3, self.z_dim], mean=0, stddev=1)
        z1 = np.random.uniform(-1, 1, (400, self.z_dim)).astype(np.float)
        self.load_model()
        gLoss = 0
        dLossFake, dLossReal = 1, 1
        d_real_count, d_fake_count, g_count = 0, 0, 0
        for i in range(500000):
            real_image_batch = self.mnist.train.next_batch(self.batch_size)[0].reshape([self.batch_size, 28, 28, 1])
            self.noise_input = np.random.uniform(-1, 1, (self.batch_size, self.z_dim)).astype(np.float)
            _, dLossReal, dLossFake, gLoss = self.sess.run([self.d_optimizer, self.d_loss_real, self.d_loss_fake, \
                                                            self.g_loss], \
                                                           {self.x: real_image_batch, \
                                                            self.z: self.noise_input})
            _, dLossReal, dLossFake, gLoss = self.sess.run([self.g_optimizer, self.d_loss_real, self.d_loss_fake, \
                                                            self.g_loss], \
                                                           {self.x: real_image_batch, self.z: self.noise_input})
            _, dLossReal, dLossFake, gLoss = self.sess.run([self.g_optimizer, self.d_loss_real, self.d_loss_fake, \
                                                            self.g_loss], \
                                                           {self.x: real_image_batch, self.z: self.noise_input})

            if i % 500 == 0:
                # Periodically display a sample image in the notebook
                # (These are also being sent to TensorBoard every 10 iterations)
                self.save_model(i, z1)

    def load_model(self):
        self.checkpoint = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if self.checkpoint and self.checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, self.checkpoint.model_checkpoint_path)
            print("Successfully loaded:", self.checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")
    def save_model(self, i, z1):
        self.saver.save(self.sess, self.checkpoint_dir+'/dcgan.ckpt', global_step=i)
        images = self.Gz.eval(session=self.sess, feed_dict = {self.z : z1})
        d_result = self.Dx.eval(session=self.sess, feed_dict ={self.x: images})
        print("TRAINING STEP", i, "AT", datetime.datetime.now())

        scipy.misc.imsave('{}/{}_gen_image.png'.format(self.result_dir,i), self.merge(images, [28, 28]))
        '''for j in range(3):
            print("Discriminator classification", d_result[j])
            im = images[j, :, :, 0]
            #plt.imsave('{}/{}_{}_gen_image.png'.format(self.result_dir,i,j),im)
            scipy.misc.imsave('{}/{}_{}_gen_image.png'.format(self.result_dir,i,j),im)'''
    def load_mnist(self):
        self.mnist = input_data.read_data_sets(self.data_dir, one_hot=True)

    def merge(self, images, size):
        h, w = images.shape[1], images.shape[2]
        img = np.zeros((int(h * size[0]), int(w * size[1]), 3))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image

        return img
