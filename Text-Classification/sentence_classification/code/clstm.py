import numpy as np
import tensorflow as tf


class TextCLSTM(object):
    """
    A C-LSTM classifier for text classification
    Reference: A C-LSTM Neural Network for Text Classification
    """
    def __init__(
        self,
        max_len,
        num_classes,
        vocab_size,
        embedding_size,
        filter_sizes,
        num_filters,
        num_layers,
        l2_reg_lambda
    ):
        self.max_len = max_len
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.hidden_size = len(self.filter_sizes) * self.num_filters
        self.num_layers = num_layers
        self.l2_reg_lambda = l2_reg_lambda

        # Placeholders
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[], name="batch_size")
        self.input_x = tf.placeholder(dtype=tf.int32, shape=[None, self.max_len], name="input_x")
        self.input_y = tf.placeholder(dtype=tf.int64, shape=[None, self.num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name="dropout_keep_prob")
        self.sequence_length = tf.placeholder(dtype=tf.int32, shape=[None], name="sequence_length")

        # L2 loss
        self.l2_loss = tf.constant(0.0)

        # Word embedding
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            self.Embedding = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0), name="embedding")
            embed = tf.nn.embedding_lookup(self.Embedding, self.input_x)
            inputs = tf.expand_dims(embed, -1)

        # Input dropout
        inputs = tf.nn.dropout(inputs, keep_prob=self.dropout_keep_prob)

        conv_outputs = []
        max_feature_length = self.max_len - max(self.filter_sizes) + 1
        # Convolutional layer with different lengths of filters in parallel
        # No max-pooling
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.variable_scope('conv-%s' % filter_size):
                # [filter size, embedding size, channels, number of filters]
                filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
                W = tf.get_variable('weights', filter_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
                b = tf.get_variable('biases', [self.num_filters], initializer=tf.constant_initializer(0.0))

                # Convolution
                conv = tf.nn.conv2d(
                    inputs,
                    W,
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='conv'
                )
                # Activation function
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')

                # Remove channel dimension
                h_reshape = tf.squeeze(h, [2])
                # Cut the feature sequence at the end based on the maximum filter length
                h_reshape = h_reshape[:, :max_feature_length, :]

                conv_outputs.append(h_reshape)

        # Concatenate the outputs from different filters
        if len(self.filter_sizes) > 1:
            rnn_inputs = tf.concat(conv_outputs, -1)
        else:
            rnn_inputs = h_reshape

        # LSTM cell
        cell = tf.contrib.rnn.LSTMCell(
            self.hidden_size,
            forget_bias=1.0,
            state_is_tuple=True,
            reuse=tf.get_variable_scope().reuse
        )
        # Add dropout to LSTM cell
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.dropout_keep_prob)

        # Stacked LSTMs
        cell = tf.contrib.rnn.MultiRNNCell([cell]*self.num_layers, state_is_tuple=True)

        self._initial_state = cell.zero_state(self.batch_size, dtype=tf.float32)

        # Feed the CNN outputs to LSTM network
        with tf.variable_scope('LSTM'):
            outputs, state = tf.nn.dynamic_rnn(
                cell,
                rnn_inputs,
                initial_state=self._initial_state,
                sequence_length=self.sequence_length
            )
            # self.final_state = state
            output_rnn = tf.concat(outputs, axis=2)
            self.output_rnn_last = tf.reduce_mean(output_rnn, axis=1)

        # Softmax output layer
        with tf.name_scope('softmax'):
            softmax_w = tf.get_variable('softmax_w', shape=[self.hidden_size, self.num_classes], dtype=tf.float32)
            softmax_b = tf.get_variable('softmax_b', shape=[self.num_classes], dtype=tf.float32)

            # L2 regularization for output layer
            self.l2_loss += tf.nn.l2_loss(softmax_w)
            self.l2_loss += tf.nn.l2_loss(softmax_b)

            # logits
            # self.logits = tf.matmul(self.final_state[self.num_layers - 1].h, softmax_w) + softmax_b
            self.logits = tf.matmul(self.output_rnn_last, softmax_w) + softmax_b
            predictions = tf.nn.softmax(self.logits)
        with tf.name_scope("prediction"):
            self.predictions = tf.argmax(predictions, 1, name='predictions')

        # Loss
        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
            self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * self.l2_loss

        # Accuracy
        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.correct_num = tf.reduce_sum(tf.cast(correct_predictions, tf.float32))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')
