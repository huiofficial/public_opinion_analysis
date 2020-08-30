import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
import data_preprocess

tf.flags.DEFINE_string("cnn_dir", "../models/cnn/", "cnn directory")
tf.flags.DEFINE_string("rnn_dir", "../models/rnn/", "rnn directory")
tf.flags.DEFINE_string("rcnn_dir", "../models/rcnn/sent5/checkpoints/", "rcnn directory")
tf.flags.DEFINE_string("clstm_dir", "../models/clstm/", "clstm directory")
tf.flags.DEFINE_string("cnn_model", "model-900", "cnn model")
tf.flags.DEFINE_string("rnn_model", "model-800", "rnn model")
tf.flags.DEFINE_string("rcnn_model", "model-2600", "rcnn model")
tf.flags.DEFINE_string("clstm_model", "model-600", "clstm model")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_string("text_dir", "../models/company/text/", "text classification directory")
tf.flags.DEFINE_string("text_model", "text_model", "text classification model")
tf.flags.DEFINE_string("sen3_dir", "../models/company/sen3/", "sentiment classification directory")
tf.flags.DEFINE_string("sen3_model", "sen3", "3 sentiment classification model")
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("sen5_dir", "../models/company/sen5/", "sentiment classification directory")
tf.flags.DEFINE_string("sen5_model", "sen5", "3 sentiment classification model")
FLAGS = tf.flags.FLAGS

def load_model(model_dir, model_name):
    graph = tf.Graph()
    vocab = load_vocab(model_dir+"vocab")
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement
        )
        sess = tf.Session(config = session_conf)
        with sess.as_default():
            saver = tf.train.import_meta_graph(model_dir+model_name+".meta")
            saver.restore(sess, model_dir+model_name)
            return graph, sess, vocab

def load_models():
    cnn_dir = tf.train.load_checkpoint(FLAGS.cnn_dir)
    rnn_dir = tf.train.load_checkpoint(FLAGS.rnn_dir)
    rcnn_dir = tf.train.load_checkpoint(FLAGS.rcnn_dir)
    clstm_dir = tf.train.load_checkpoint(FLAGS.clstm_dir)
    cnn_graph, cnn_sess, cnn_vocab = load_model(cnn_dir, FLAGS.cnn_model)
    rnn_graph, rnn_sess, rnn_vocab = load_model(rnn_dir, FLAGS.rnn_model)
    rcnn_graph, rcnn_sess, rcnn_vocab = load_model(rcnn_dir, FLAGS.rcnn_model)
    clstm_graph, clstm_sess, clstm_vocab = load_model(clstm_dir, FLAGS.clstm_model)
    return cnn_graph, cnn_sess, cnn_vocab, rnn_graph, rnn_sess, rnn_vocab, rcnn_graph, rcnn_sess, rcnn_vocab, clstm_graph, clstm_sess, clstm_vocab

def load_vocab(vocab_path):
    return learn.preprocessing.VocabularyProcessor.restore(vocab_path)

def formate_str(sentence, vocab):
    sentence = data_preprocess.clean_str(sentence)
    sentence = [sentence]
    sent_idx = np.array(list(vocab.fit_transform(sentence)))
    return sent_idx

def _classification(model_dir, model_file, y):
    _ = tf.train.load_checkpoint(model_dir)
    graph, sess, vocab = load_model(model_dir, model_file)
    input_x = graph.get_operation_by_name("input_x").outputs[0]
    input_y = graph.get_operation_by_name("input_y").outputs[0]
    dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
    accuracy = graph.get_operation_by_name("accuracy/accuracy").outputs[0]
    predictions = graph.get_operation_by_name("prediction/predictions").outputs[0]
    while True:
        sentence = input("please input a sentence in Chinese, MAX LENGTH=100\n")
        formated_sentence_idx = formate_str(sentence, vocab)
        pred_y, acc = sess.run([predictions, accuracy], {
            input_x: formated_sentence_idx,
            input_y: y,
            dropout_keep_prob: 1.0
        })
        print(pred_y)

def text_classification():
    _classification(FLAGS.text_dir, FLAGS.text_model, [[0,0,0,0,0]])

def sen3_classification():
    _classification(FLAGS.sen3_dir, FLAGS.sen3_model, [[0,0,0]])

def sen5_classification():
    _classification(FLAGS.sen5_dir, FLAGS.sen5_model, [[0,0,0,0,0]])
    