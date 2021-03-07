import tensorflow as tf
import numpy as np
from .TDNN import TDNN

class CNN():

    def __init__(self,
            general_utils,batch_size=15,
            char_hidden_size=100,word_hidden_size=100,max_word_length=60,
            max_seq_length=60,feature_maps=[50, 100, 150, 200, 200, 200, 200],
            kernels=[1,2,3,4,5,6,7]):

        self.init = None
        self.data = general_utils

        self.batch_size  = batch_size

        self.char_num     = self.data.char_num
        self.word_num     = self.data.word_num
        self.sentence_num = self.data.sentence_num

        self.char_hidden_size = char_hidden_size
        self.word_hidden_size = word_hidden_size

        self.char_inputs = char_hidden_size
        self.word_inputs = word_hidden_size

        self.max_word_length = max_word_length
        self.max_seq_length  = max_seq_length

        self.feature_maps = feature_maps
        self.kernels = kernels

    def BuildModel(self):
        
        ex_sentence = "world ! my name is computer"

        weight = tf.constant([[[[1.]],[[1.]]],  # filter 생성
                              [[[1.]],[[1.]]]])

        with tf.variable_scope("CNN"):
            char_weight = tf.get_variable(
                "char_embedding",[self.char_num,self.char_hidden_size])
            word_weight = tf.get_variable(
                "word_embedding",[self.word_num,self.word_hidden_size])

            with tf.variable_scope("CNN_scope") as scope:
                self.char_inputs = tf.placeholder(
                    tf.int32,[self.batch_size, self.max_seq_length,self.max_word_length])
                self.word_inputs = tf.placeholder(
                    tf.int32,[self.batch_size, self.max_seq_length])

                char_indices = tf.split(self.char_inputs, self.max_seq_length,1)
                word_indices = tf.split(self.word_inputs, self.max_seq_length, tf.expand_dims(self.word_inputs, -1))

                for idx in range(self.max_seq_length):
                    char_idx = tf.reshape(char_indices[idx],[-1,self.max_seq_length])
                    word_idx = tf.reshape(word_indices[idx],[-1,1])

                    char_embed = tf.nn.embedding_lookup(char_weight, char_idx)
                    print("char_embed shape : ", tf.shape(char_embed))

                    idx_sentence = np.array(self.sentence2idx(ex_sentence), dtype=np.float32)
                    print(idx_sentence)

                    conv2d = tf.nn.conv2d(idx_sentence, weight, strides=[1,1,1,1], padding='SAME')
                    print("conv2d shape : ", tf.shape(conv2d))

                    # conv2d_img = conv2d.eval()
                    # char_cnn = TDNN(self.char_hidden_size, self.feature_maps, self.kernels)

                    word_embed = tf.nn.embedding_lookup(word_weight, word_idx)

                    print("word_embed shape : ", tf.shape(word_embed))
                    cnn_output = tf.concat(1, [char_cnn.output, tf.squeeze(word_embed, [1])])

    def sentence2idx(self,input):
        return_value = []
        for word in input.split(" "):
            return_value.append(self.data.word2idx[word])
        return return_value

    def cv_cnn(self,hidden_size,feature_maps,kernels):
        return
