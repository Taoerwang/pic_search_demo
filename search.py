from milvus import Milvus, IndexType, MetricType, Status
import random

import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.utils import np_utils
import matplotlib.pyplot as plt
import os
from keras import backend as K
import pandas as pd
import os
import cv2

class VAE:
    model_name = "VAE"
    paper_name = "Auto-Encoding Variational Bayes(变分自编码器)"
    paper_url = "https://arxiv.org/abs/1312.6114"
    data_sets = "MNIST and Fashion-MNIST"

    def __init__(self, data_name):
        self.data_name = data_name
        self.img_counts = 60000
        self.img_rows = 28
        self.img_cols = 28
        self.dim = 1
        self.noise_dim = 100

        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = self.load_data()
        # self.train_images = np.reshape(self.train_images, (-1, self.img_rows * self.img_cols)) / 255
        # self.test_images = np.reshape(self.test_images, (-1, self.img_rows * self.img_cols)) / 255
        self.train_labels_one_dim = self.train_labels
        self.test_labels_one_dim = self.test_labels
        self.train_labels = np_utils.to_categorical(self.train_labels)
        self.test_labels = np_utils.to_categorical(self.test_labels)

    def load_data(self):
        if self.data_name == "fashion_mnist":
            data_sets = keras.datasets.fashion_mnist
        elif self.data_name == "mnist":
            data_sets = keras.datasets.mnist
        else:
            data_sets = keras.datasets.mnist
        return data_sets.load_data()

    def encoder(self, e):
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):

            e = tf.layers.dense(e, 512, tf.nn.relu)
            e = tf.layers.batch_normalization(e)
            e = tf.layers.dense(e, 256, tf.nn.relu)
            e = tf.layers.batch_normalization(e)
            out_mean = tf.layers.dense(e, 2)
            out_stddev = tf.layers.dense(e, 2, )
            return out_mean, out_stddev

    def decoder(self, d):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            d = tf.layers.dense(d, 128, tf.nn.relu)
            d = tf.layers.batch_normalization(d)
            d = tf.layers.dense(d, 256, tf.nn.relu)
            d = tf.layers.batch_normalization(d)
            d = tf.layers.dense(d, 512, tf.nn.relu)
            d = tf.layers.batch_normalization(d)
            out = tf.layers.dense(d, 784, tf.nn.sigmoid)  # 数据归一化， 0-1
            return out

    def build_model(self, learning_rate=0.0002):

        x_real = tf.placeholder(tf.float32, [None, self.img_rows * self.img_cols])
        z_noise = tf.placeholder(tf.float32, [None, 2])

        z_mean, z_log_var = self.encoder(x_real)
        # 我们选择拟合logσ2而不是直接拟合σ2，是因为σ2总是非负的，
        # 需要加激活函数处理，而拟合logσ2不需要加激活函数，因为它可正可负。
        # guessed_z = z_mean + tf.exp(z_log_stddev2 / 2) * samples

        epsilon = K.random_normal(shape=K.shape(z_mean))  # 默认N（0,1）

        guessed_z = z_mean + K.exp(z_log_var / 2) * epsilon  # 编码器和解码器之间的对抗

        x_fake = self.decoder(guessed_z)

        z_real = self.decoder(z_noise)  # 观察隐藏空间的时候使用

        xent_loss = K.sum(K.binary_crossentropy(x_real, x_fake), axis=-1)
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        cost = K.mean(xent_loss + kl_loss)
        t_vars = tf.trainable_variables()
        e_d_vars = [var for var in t_vars if 'encoder' or 'decoder' in var.name]
        optimizer = tf.train.AdamOptimizer(0.001).minimize(cost, var_list=e_d_vars)

        return x_real, x_fake, cost, optimizer, z_noise, z_real, guessed_z

    def train(self, train_steps=100000, batch_size=100, learning_rate=0.001, save_model_numbers=3):
        x_real, x_fake, cost, optimizer, z_noise, z_real, guessed_z = self.build_model(learning_rate)
        saver = tf.train.Saver(max_to_keep=save_model_numbers)
        if not os.path.exists('out/'):
            os.makedirs('out/')

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # merged_summary_op = tf.summary.merge_all()
            # summary_writer = tf.summary.FileWriter('log/mnist_with_summaries', sess.graph)
            for i in range(train_steps):
                batch_index = np.random.randint(0, self.img_counts, batch_size)
                batch_real = self.train_images[batch_index]

                sess.run(optimizer, feed_dict={x_real: batch_real})

                if i % 1000 == 0:

                    auto_encoder_loss_curr = sess.run(cost, feed_dict={x_real: batch_real})

                    print('step: ' + str(i))
                    print('D_loss: ' + str(auto_encoder_loss_curr))
                    print()
                    saver.save(sess, 'ckpt/mnist.ckpt', global_step=i)

                    x_fake_ = sess.run(x_fake, feed_dict={x_real: batch_real})

                    r, c = 10, 10
                    fig, axs = plt.subplots(r, c)
                    cnt = 0
                    for p in range(r):
                        for q in range(c):
                            axs[p, q].imshow(np.reshape(batch_real[cnt], (28, 28)), cmap='gray')
                            axs[p, q].axis('off')
                            cnt += 1
                    fig.savefig("out/%d_real.png" % i)
                    plt.close()

                    r, c = 10, 10
                    fig, axs = plt.subplots(r, c)
                    cnt = 0
                    for p in range(r):
                        for q in range(c):
                            axs[p, q].imshow(np.reshape(x_fake_[cnt], (28, 28)), cmap='gray')
                            axs[p, q].axis('off')
                            cnt += 1
                    fig.savefig("out/%d_fake.png" % i)
                    plt.close()

                    test_z = sess.run(guessed_z, feed_dict={x_real: self.train_images})
                    fig = plt.figure()
                    ax = fig.add_subplot(1, 1, 1)
                    ax.scatter(test_z[:, 0], test_z[:, 1], c=self.train_labels_one_dim, s=1)
                    fig.savefig("out/%d_prediction.png" % i)
                    plt.close()

    def search(self, img_id):

        x_real, x_fake, cost, optimizer, z_noise, z_real, guessed_z = self.build_model()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            model_file = tf.train.latest_checkpoint('D:/py_project/untitled1/polls/homework/vae/ckpt/')
            # 加载参数
            saver.restore(sess, model_file)

            # img_id = 'test_img_0.png'
            search_img_path = "D:/py_project/untitled1/polls/static/polls/homework/"+img_id
            search_img = cv2.imread(search_img_path, cv2.IMREAD_GRAYSCALE)
            search_img = np.reshape(search_img, [-1, 784])/255

            # 获取 要搜索图片的特征向量

            test_z = sess.run(guessed_z, feed_dict={x_real: search_img})

            print(img_id, test_z)

            # 连接milvus数据库 并进行查询
            milvus = Milvus()
            milvus.connect(host='localhost', port='19530')
            collection_name = 'mnist'
            search_param = {'nprobe': 16}
            status, result = milvus.search(collection_name=collection_name, query_records=test_z, top_k=100,
                                           params=search_param)
            milvus.disconnect()

            # index = []
            # for row in result:
            #     for item in row:
            #         index.append(item.id)
            # batch_real = self.train_images[index]
            # r, c = 10, 10
            # fig, axs = plt.subplots(r, c)
            # cnt = 0
            # for p in range(r):
            #     for q in range(c):
            #         axs[p, q].imshow(np.reshape(batch_real[cnt], (28, 28)), cmap='gray')
            #         axs[p, q].axis('off')
            #         cnt += 1
            # print(os.path)
            # fig.savefig("D:/py_project/untitled1/polls/static/polls/homework/d_real.png")
            # plt.show()
            # plt.close()

            # 保存查询结果
            for row in result:
                for i, item in enumerate(row):
                    # a = 1
                    # print(item.id)
                    result_img_path = "D:/py_project/untitled1/polls/static/polls/homework/search_result/result_"
                    batch_test_pic = self.train_images[item.id]
                    pic_name = result_img_path + str(i) + ".png"
                    cv2.imwrite(pic_name, batch_test_pic)

            print('search success!')

            return True


# if __name__ == '__main__':
#     data = ['fashion_mnist', 'mnist']
#     model = VAE(data[1])
#     # model.train()
#     model.search('test_img_0.png')
