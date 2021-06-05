import tensorflow as tf
import ops
import numpy as np

class NextItNet_Decoder:

    def __init__(self, model_para):
        self.model_para = model_para
        self.load_model = model_para['load_model']
        self.method = model_para['method']
        self.L2 = model_para['L2']
        embedding_width =  model_para['dilated_channels']

        
        if self.load_model:
            self.model_path = model_para['model_path']
            self.reader = tf.train.NewCheckpointReader(self.model_path)
            variable_name = 'allitem_embeddings'
            initial_value = self.get_parameters(self.reader, variable_name, variable_name)
            self.allitem_embeddings = tf.get_variable('allitem_embeddings',
                                                    [model_para['item_size'], embedding_width],
                                                    initializer=tf.constant_initializer(initial_value, verify_shape=True))
        else:
            self.allitem_embeddings = tf.get_variable('allitem_embeddings',
                                                    [model_para['item_size'], embedding_width],
                                                    initializer=tf.truncated_normal_initializer(stddev=0.02))

    def get_parameters(self, reader, variable_name, new_variable_name):
        print(variable_name, "  --->  ", new_variable_name)
        param = reader.get_tensor(variable_name)
        return param
        

    def train_graph(self):
        self.itemseq_input = tf.placeholder('int32',
                                         [None, None], name='itemseq_input')
        label_seq, self.dilate_input=self.model_graph(self.itemseq_input, train=True)

        model_para = self.model_para

        logits_2D = tf.reshape(self.dilate_input, [-1,model_para['dilated_channels']])
        

        self.softmax_w = tf.get_variable("softmax_w", [model_para['item_size'],  model_para['dilated_channels']], tf.float32, tf.random_normal_initializer(0.0, 0.01))
        self.softmax_b = tf.get_variable("softmax_b", [model_para['item_size']], tf.float32, tf.constant_initializer(0.1))

        label_flat = tf.reshape(label_seq, [-1, 1])
        num_sampled = int(0.2 * model_para['item_size'])

        loss = tf.nn.sampled_softmax_loss(self.softmax_w, self.softmax_b, label_flat, logits_2D, num_sampled, model_para['item_size'])

     
        self.loss = tf.reduce_mean(loss)

        if self.L2 != 0:
            regularization = self.L2 * tf.reduce_mean([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
            self.loss = self.loss + regularization


    def model_graph(self, itemseq_input, train=True):
        model_para = self.model_para
        context_seq = itemseq_input[:, 0:-1]
        label_seq = itemseq_input[:, 1:]

        self.context_embedding = tf.nn.embedding_lookup(self.allitem_embeddings,
                                                        context_seq, name="context_embedding")
        dilate_input = self.context_embedding
        layer_num = len(model_para['dilations'])

        for layer_id, dilation in enumerate(model_para['dilations']):
            if self.load_model:
                dilate_input = ops.nextitnet_residual_block_alpha(dilate_input, dilation,
                                                                layer_id, self.method, model_para['dilated_channels'],
                                                                model_para['kernel_size'], self.reader, layer_num, train=train)
            else:
                dilate_input = ops.nextitnet_residual_block_alpha(dilate_input, dilation,
                                                                layer_id, self.method, model_para['dilated_channels'],
                                                                model_para['kernel_size'], None, layer_num, train=train)

        return label_seq, dilate_input


    def predict_graph(self, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        self.input_predict = tf.placeholder('int32', [None, None], name='input_predict')

        label_seq, dilate_input = self.model_graph(self.input_predict, train=False)
        model_para = self.model_para


        logits_2D = tf.reshape(dilate_input[:, -1:, :], [-1, model_para['dilated_channels']])
        logits_2D = tf.matmul(logits_2D, tf.transpose(self.softmax_w))
        logits_2D = tf.nn.bias_add(logits_2D, self.softmax_b)
        

        probs_flat = tf.nn.softmax(logits_2D)

        self.g_probs = tf.reshape(probs_flat, [-1, 1, model_para['item_size']])
        self.top_10 = tf.nn.top_k(self.g_probs, 10)
        self.top_5 = tf.nn.top_k(self.g_probs, 5)






















