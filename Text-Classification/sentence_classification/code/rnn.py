import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

class TextRNN(object):
    def __init__(
        self, 
        sequence_length, 
        num_classes, 
        vocab_size, 
        embedding_size,
        learning_rate, 
        batch_size, 
        decay_steps, 
        decay_rate,  
        is_training, 
        initializer=tf.random_normal_initializer(stddev=0.1)
    ):
        # set hyperparamter
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.sequence_length=sequence_length
        self.vocab_size=vocab_size
        # print("==============vocab_size: "+str(vocab_size))
        self.embedding_size=embedding_size
        self.hidden_size=embedding_size
        self.is_training=is_training
        self.learning_rate=learning_rate
        self.initializer=initializer

        self.num_sampled = 20 # ???

        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.int32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32,name="dropout_keep_prob")

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.decay_steps, self.decay_rate = decay_steps, decay_rate

        self.instantiate_weights()
        self.logits = self.inference()
        if not is_training:
            return
        self.loss = self.loss()
        self.train_op = self.train()
        with tf.name_scope("prediction"):
            self.predictions = tf.argmax(self.logits, axis=1, name="predictions")

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name = "accuracy")
        # correct_prediction = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
        # self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")

    def instantiate_weights(self):
        """define all weights here"""
        # embedding matrix
        with tf.name_scope("embedding"): 
            # [vocab_size,embedding_size] tf.random_uniform([self.vocab_size, self.embedding_size],-1.0,1.0)
            self.Embedding = tf.get_variable("Embedding", shape=[self.vocab_size, self.embedding_size],initializer=self.initializer)
            # [embedding_size,label_size]
            self.W_projection = tf.get_variable("W_projection",shape=[self.hidden_size*2, self.num_classes],initializer=self.initializer)
            # [label_size]
            self.b_projection = tf.get_variable("b_projection",shape=[self.num_classes])
    
    def inference(self):
        # 1.get emebedding of words in the sentence
        self.embedded_words = tf.nn.embedding_lookup(self.Embedding, self.input_x)
        # 2. BiLSTM
        lstm_fw_cell = rnn.BasicLSTMCell(self.hidden_size)
        lstm_bw_cell = rnn.BasicLSTMCell(self.hidden_size)
        if self.dropout_keep_prob is not None:
            lstm_fw_cell = rnn.DropoutWrapper(lstm_fw_cell, input_keep_prob=self.dropout_keep_prob)
            lstm_bw_cell = rnn.DropoutWrapper(lstm_bw_cell, input_keep_prob=self.dropout_keep_prob)
        # (outputs=(fw_output, bw_output), output_states)
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, self.embedded_words, dtype=tf.float32)
        print("outputs:===>", outputs)
        # [batch_size, sequence_length, hidden_size*2]
        # 3. concat output
        output_rnn = tf.concat(outputs, axis=2)
        # output_rnn = tf.concat(concat_dim=2, outputs)
        # [batch_size, hidden_size*2]
        self.output_rnn_last = tf.reduce_mean(output_rnn, axis=1)
        print("output_rnn_last:===>", self.output_rnn_last)
        # 4. linear layer for logits
        with tf.name_scope("output"):
            logits = tf.matmul(self.output_rnn_last, self.W_projection)+self.b_projection
        return logits

    def loss(self, l2_lambda = 0.0001):
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
            loss = tf.reduce_mean(losses)
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if "bias" not in v.name])*l2_lambda
            loss = loss+l2_losses
        return loss

    def loss_nce(self, l2_lambda = 0.001):
        if self.is_training:
            labels = tf.expand_dims(self.input_y, 1)
            loss = tf.reduce_mean(tf.nn.nce_loss(
                weights=tf.transpose(self.W_projection),
                biases=self.b_projection,
                labels=labels,
                inputs=self.output_rnn_last,
                num_sampled=self.num_sampled,
                num_classes=self.num_classes,
                partition_strategy="div")
            )
        l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if "bias" not in v.name])*l2_lambda
        loss = loss+l2_losses
        return loss

    def train(self):
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps, self.decay_rate, staircase=True)
        train_op = tf.contrib.layers.optimize_loss(self.loss, global_step=self.global_step, learning_rate=self.learning_rate, optimizer="Adam")
        return train_op

#test started
def test():
    #below is a function test; if you use this for text classifiction, you need to tranform sentence to indices of vocabulary first. then feed data to the graph.
    num_classes=10
    learning_rate=0.01
    batch_size=8
    decay_steps=1000
    decay_rate=0.9
    sequence_length=5
    vocab_size=10000
    embedding_size=100
    is_training=True
    dropout_keep_prob=1#0.5
    textRNN=TextRNN(num_classes, learning_rate, batch_size, decay_steps, decay_rate,sequence_length,vocab_size,embedding_size,is_training)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(100):
            input_x=np.zeros((batch_size,sequence_length)) #[None, self.sequence_length]
            input_y=input_y=np.array([1,0,1,1,1,2,1,1]) #np.zeros((batch_size),dtype=np.int32) #[None, self.sequence_length]
            loss,acc,predict,_=sess.run([textRNN.loss,textRNN.accuracy,textRNN.predictions,textRNN.train_op],feed_dict={textRNN.input_x:input_x,textRNN.input_y:input_y,textRNN.dropout_keep_prob:dropout_keep_prob})
            print("loss:",loss,"acc:",acc,"label:",input_y,"prediction:",predict)
# test()       
