import tensorflow as tf
import numpy as np
import os
import sys
import pickle
from sklearn import metrics
import data_preprocess
from tensorflow.contrib import learn

"""
数据文件
"""
tf.flags.DEFINE_string("mr_test_review_file", "../data/formated_data/mr/test_x.pkl", "review source for the test data.")
tf.flags.DEFINE_string("mr_test_tag_file", "../data/formated_data/mr/test_y.pkl", "tag source for the test data.")
tf.flags.DEFINE_string("sst1_x", "../data/formated_data/sst1/test_x.pkl", "review source for the test data.")
tf.flags.DEFINE_string("sst1_y", "../data/formated_data/sst1/test_y.pkl", "tag source for the test data.")
tf.flags.DEFINE_string("company_x", "../data/formated_data/company/sen5/test_x.pkl", "review source for the test data.")
tf.flags.DEFINE_string("company_y", "../data/formated_data/company/sen5/test_y.pkl", "tag source for the test data.")
"""
模型参数
"""
tf.flags.DEFINE_string("checkpoint_dir", "runs/1529472388/checkpoints/", "Checkpoint directory from training run")
tf.flags.DEFINE_string("vocab_path", "runs/1529472388/vocab", "Checkpoint directory from training run")

"""
模型选择
"""
tf.flags.DEFINE_string("model", "cnn", "selected model of implementations, candidates: {cnn, rnn, rcnn, clstm}")

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

with open(FLAGS.company_x, "rb") as file:
    test_x = pickle.load(file)
with open(FLAGS.company_y, "rb") as file:
    test_y = pickle.load(file)

vocab_processor = learn.preprocessing.VocabularyProcessor.restore(FLAGS.vocab_path)

test_x = np.array(list(vocab_processor.fit_transform(test_x)))
test_y = np.array(test_y)
test_lengths = [len(test_x[i].tolist()) for i in range(len(test_x))]
# vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
# vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
# test_x = np.array(list(vocab_processor.fit_transform(test_x)))

checkpoint_file = tf.train.load_checkpoint(FLAGS.checkpoint_dir)

graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config = session_conf)
    with sess.as_default():
        # saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        # saver.restore(sess, checkpoint_file)
        saver = tf.train.import_meta_graph(FLAGS.checkpoint_dir+"model-900.meta")
        saver.restore(sess, FLAGS.checkpoint_dir+"model-900")

        input_x = graph.get_operation_by_name("input_x").outputs[0]
        input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        accuracy = graph.get_operation_by_name("accuracy/accuracy").outputs[0]
        predictions = graph.get_operation_by_name("prediction/predictions").outputs[0]

        if FLAGS.model == "clstm":
            batch_size = graph.get_operation_by_name("batch_size").outputs[0]
            sequence_length = graph.get_operation_by_name("sequence_length").outputs[0]
            pred_y, acc = sess.run([predictions, accuracy], {
                input_x: test_x,
                input_y: test_y,
                batch_size: len(test_x),
                sequence_length: test_lengths,
                dropout_keep_prob: 1.0
            })
        else:
            pred_y, acc = sess.run([predictions, accuracy], {
                input_x: test_x,
                input_y: test_y,
                dropout_keep_prob: 1.0
            })

        # batches = data_preprocess.batch_iter(list(test_x), FLAGS.batch_size, 1, shuffle = False)
        # predictions_result = []
        # for batch_x in batches:
        #     batch_y = sess.run(predictions, {input_x: batch_x, dropout_keep_prob: 1.0})
        #     predictions_result = np.concatenate([predictions_result, batch_y])
# print(np.argmax(test_y, 1))
# print(pred_y)
# print(acc)

test_y = np.argmax(test_y, 1).tolist()
# pred_y = test_y.tolist()
# target_names = ["无关分类", "物业服务质量相关", "物业社会活动相关", "物业安全事件相关", "地产房产各类活动"]
# target_names = ["负面", "中性", "正面"]
target_names = ["很差", "较差", "中性", "较好", "很好"]
res = metrics.classification_report(test_y, pred_y, target_names=target_names)
print(res)
with open(os.path.join(FLAGS.checkpoint_dir, "..", "predictions.txt"), "w") as file:
    file.write(str(res))
    for prediction in pred_y:
        file.write(str(int(prediction)+1)+"\n")
