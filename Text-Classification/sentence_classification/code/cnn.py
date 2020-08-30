import tensorflow as tf
import numpy as np


class TextCNN(object):
    """
    filter 指模板
    filter_sizes 候选的模板大小
    num_filters 每个 filter 的数量
    """
    def __init__(
        self, 
        sequence_length, 
        num_classes, 
        vocab_size, 
        embedding_size, 
        filter_sizes, 
        num_filters, 
        l2_reg_lambda = 0.0
    ):
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        l2_loss = tf.constant(l2_reg_lambda)

        with tf.device('/gpu:0'), tf.name_scope("embedding"):
            self.Embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="W")
            """
            embedded_chars 将输入查表得到对应的 embedding 表示，shape=[batch_size, sequence_length, embedding_size]
            embedded_chars_expanded，shape=[batch_size, sequence_length, embedding_size, 1]
            """
            self.embedded_chars = tf.nn.embedding_lookup(self.Embedding, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
        
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                """
                卷积层
                filter 为 filter_size*embedding_size 的模板
                strides 的每个维度表示每次 filter 在各个维度的移动范围
                """
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides = [1, 1, 1, 1],
                    padding = "VALID",
                    name = "conv"
                )
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                pooled = tf.nn.max_pool(
                    h,
                    ksize = [1, sequence_length-filter_size+1, 1, 1],
                    strides = [1, 1, 1, 1],
                    padding = "VALID",
                    name = "pool"
                )
                pooled_outputs.append([pooled])
        
        num_filters_total = num_filters*len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
        
        # 输出
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape = [num_classes]), name = "b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name = "scores")

        with tf.name_scope("prediction"):
            self.predictions = tf.argmax(self.scores, 1, name = "predictions")
        
        # 损失函数
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits = self.scores, labels = self.input_y)
            self.loss = tf.reduce_mean(losses)+l2_reg_lambda*l2_loss

        # 准确率
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name = "accuracy")

