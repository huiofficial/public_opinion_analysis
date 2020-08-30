import tensorflow as tf
import numpy as np
import _pickle as cPickle
import pandas as pd
import os
import time
import sys
import gensim
import datetime
import data_preprocess
from tensorflow.contrib import learn

from cnn import TextCNN
from rnn import TextRNN
from rcnn import TextRCNN
from clstm import TextCLSTM

"""
数据文件
"""
tf.flags.DEFINE_string("mr_train_review_file", "../data/formated_data/mr/train_x.pkl", "review source for the train data.")
tf.flags.DEFINE_string("mr_train_tag_file", "../data/formated_data/mr/train_y.pkl", "tag source for the train data.")
tf.flags.DEFINE_string("mr_dev_review_file", "../data/formated_data/mr/dev_x.pkl", "review source for the dev data.")
tf.flags.DEFINE_string("mr_dev_tag_file", "../data/formated_data/mr/dev_y.pkl", "tag source for the dev data.")
tf.flags.DEFINE_string("word2vec", "./GoogleNews-vectors-negative300.bin", "use word2vec")

"""
模型选择
"""
tf.flags.DEFINE_string("model", "cnn", "selected model of implementations, candidates: {cnn, rnn, rcnn, clstm}")

"""
模型超参
"""
tf.flags.DEFINE_integer("embedding_size", 300, "Dimensionality of character embedding (default: 300)")
tf.flags.DEFINE_integer("context_embedding_size", 512, "Dimensionality of context embedding(= RNN state size)  (Default: 512)")
tf.flags.DEFINE_integer("hidden_size", 512, "Size of hidden layer (Default: 512)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 150, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 1.0, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_integer('num_layers', 1, 'Number of the LSTM cells')
tf.flags.DEFINE_string("cell_type", "lstm", "Type of RNN cell. Choose 'vanilla' or 'lstm' or 'gru' (Default: vanilla)")
"""
训练参数
"""
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 20, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 4, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_float("learning_rate", 1e-3, "learning rate (default:1e-3)")
tf.flags.DEFINE_integer("decay_steps", 12000, "how many steps before decay learning rate.")
tf.flags.DEFINE_float("decay_rate", 0.9, "Rate of decay for learning rate.")
tf.flags.DEFINE_boolean("is_training",True,"is traning.true:tranining,false:testing/inference")

"""
配置 session 的一些参数
"""
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
# FLAGS(sys.argv)
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

"""
加载训练数据
"""
# 公司标注数据集加载
# train_x, train_y, dev_x, dev_y, test_x, test_y = data_preprocess.load_company_data("text")
# train_x, train_y, dev_x, dev_y, test_x, test_y = data_preprocess.load_company_data("sen3")
train_x, train_y, dev_x, dev_y, test_x, test_y = data_preprocess.load_company_data("sen5")

# SST1 数据集加载
# train_x, train_y, dev_x, dev_y, test_x, test_y = data_preprocess.load_train_dev_test("sst1")
max_document_length = max([len(i.split(" ")) for i in train_x])
# max_document_length = 30
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
train_x = np.array(list(vocab_processor.fit_transform(train_x)))
dev_x = np.array(list(vocab_processor.transform(dev_x)))
test_x = np.array(list(vocab_processor.transform(test_x)))
train_y = np.array(train_y)
dev_y = np.array(dev_y)
test_y = np.array(test_y)
train_x, train_y = data_preprocess.shuffle_data(train_x, train_y)
dev_x, dev_y = data_preprocess.shuffle_data(dev_x, dev_y)
test_x, test_y = data_preprocess.shuffle_data(test_x, test_y)

# MR 数据集加载
# text_x, y = data_preprocess.load_data_and_labels("mr")
# Build vocabulary
# max_document_length = max([len(i.split(" ")) for i in text_x])
# vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
# x = np.array(list(vocab_processor.fit_transform(text_x)))
# train_x, train_y, dev_x, dev_y, test_x, test_y = data_preprocess.get_train_dev_test(x, y, "mr")

train_lengths = [len(train_x[i].tolist()) for i in range(len(train_x))]
dev_lengths = [len(dev_x[i].tolist()) for i in range(len(dev_x))]
test_lengths = [len(test_x[i].tolist()) for i in range(len(test_x))]
print("Max Length: {:d}".format(max_document_length))
print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_))) # 15372
print("Train/Dev split: {:d}/{:d}".format(len(train_y), len(dev_y))) # 8546/1098

"""
训练
"""
def model_selection(model_name):
    if model_name == "cnn":
        return TextCNN(
            sequence_length=train_x.shape[1],
            num_classes=train_y.shape[1],
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=FLAGS.embedding_size,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda
        )
    elif model_name == "rnn":
        return TextRNN(
            sequence_length=max_document_length,
            num_classes=train_y.shape[1],
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=FLAGS.embedding_size,
            learning_rate=FLAGS.learning_rate,
            batch_size=FLAGS.batch_size,
            decay_steps=FLAGS.decay_steps,
            decay_rate=FLAGS.decay_rate,
            is_training=FLAGS.is_training
        )
    elif model_name == "rcnn":
        return TextRCNN(
            sequence_length=train_x.shape[1],
            num_classes=train_y.shape[1], 
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=FLAGS.embedding_size,
            context_embedding_size=FLAGS.context_embedding_size,
            cell_type=FLAGS.cell_type,
            hidden_size=FLAGS.hidden_size,
            l2_reg_lambda=FLAGS.l2_reg_lambda
        )
    elif model_name == "clstm":
        return TextCLSTM(
            max_len=max_document_length,
            num_classes=train_y.shape[1], 
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=FLAGS.embedding_size,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            num_layers=FLAGS.num_layers,
            l2_reg_lambda=FLAGS.l2_reg_lambda
        )
    else:
        raise NotImplementedError("%s is not implemented"%(model_name))

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config = session_conf)
    with sess.as_default():
        model = model_selection(FLAGS.model)

        """
        训练过程
        """
        global_step = tf.Variable(0, name = "global_step", trainable = False)
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        grads_and_vars = optimizer.compute_gradients(model.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step = global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", model.loss)
        acc_summary = tf.summary.scalar("accuracy", model.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Write vocabulary
        vocab_processor.save(os.path.join(out_dir, "vocab"))

        sess.run(tf.global_variables_initializer())

        if FLAGS.word2vec:
            # initial matrix with random uniform
            initW = np.random.uniform(-0.25, 0.25,(len(vocab_processor.vocabulary_), FLAGS.embedding_size))
            # load any vectors from the word2vec
            print("Load word2vec file {}\n".format(FLAGS.word2vec))
            word_vectors = gensim.models.keyedvectors.KeyedVectors.load_word2vec_format(FLAGS.word2vec, binary = True)
            for word in word_vectors.wv.vocab:
                index = vocab_processor.vocabulary_.get(word)
                if index != 0:
                    initW[index] = np.array(word_vectors[word])   

            sess.run(model.Embedding.assign(initW))

            del word_vectors


        def train_step(batch_x, batch_y, batch_lengths):
            if FLAGS.model == "clstm":
                feed_dict = {
                    model.input_x: batch_x,
                    model.input_y: batch_y,
                    model.dropout_keep_prob: FLAGS.dropout_keep_prob,
                    model.batch_size: len(batch_x),
                    model.sequence_length: batch_lengths
                }
            else:
                feed_dict = {
                    model.input_x: batch_x,
                    model.input_y: batch_y,
                    model.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, model.loss, model.accuracy], 
                feed_dict = feed_dict
            )
            time_str = datetime.datetime.now().isoformat()
            # print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(batch_x, batch_y, batch_lengths, writer = None):
            if FLAGS.model == "clstm":
                feed_dict = {
                    model.input_x: batch_x,
                    model.input_y: batch_y,
                    model.dropout_keep_prob: 1.0,
                    model.batch_size: len(batch_x),
                    model.sequence_length: batch_lengths
                }
            else:
                feed_dict = {
                    model.input_x: batch_x,
                    model.input_y: batch_y,
                    model.dropout_keep_prob: 1.0
                }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, model.loss, model.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        batches = data_preprocess.batch_iter(list(zip(train_x, train_y, train_lengths)), FLAGS.batch_size, FLAGS.num_epochs, train_lengths)
        for batch in batches:
            batch_x, batch_y, batch_lengths = zip(*batch)
            train_step(batch_x, batch_y, batch_lengths)
            curr_step = tf.train.global_step(sess, global_step)
            if curr_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(dev_x, dev_y, dev_lengths, writer=dev_summary_writer)
                print("")
            if curr_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step = curr_step)
                print("Saved model checkpoint to {}\n".format(path))
