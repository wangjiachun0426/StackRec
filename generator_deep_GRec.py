import tensorflow as tf
import ops
import numpy as np
import utils_GRec


class NextItNet_Decoder:
    def __init__(self, model_para):
        self.model_para = model_para
        self.load_model = model_para['load_model']
        self.method = model_para['method']
        self.L2 = model_para['L2']
        embedding_width = model_para['dilated_channels']

        if self.load_model:
            self.model_path = model_para['model_path']
            self.reader = tf.train.NewCheckpointReader(self.model_path)
            variable_name = 'allitem_embeddings'
            initial_value = self.get_parameters(self.reader, variable_name, variable_name)
            self.allitem_embeddings = tf.get_variable('allitem_embeddings',
                                                      [model_para['item_size'], embedding_width],
                                                      initializer=tf.constant_initializer(initial_value,
                                                                                          verify_shape=True))
        else:
            self.allitem_embeddings = tf.get_variable('allitem_embeddings',
                                                      [model_para['item_size'], embedding_width],
                                                      initializer=tf.truncated_normal_initializer(stddev=0.02))

        self.itemseq_input = tf.placeholder('int32',
                                            [None, None], name='itemseq_input')


        self.softmax_w = tf.get_variable("softmax_w", [model_para['item_size'], model_para['dilated_channels']],
                                         tf.float32, tf.random_normal_initializer(0.0, 0.01))


        self.softmax_b = tf.get_variable("softmax_b", [model_para['item_size']], tf.float32,
                                         tf.constant_initializer(0.1))

    def get_parameters(self, reader, variable_name, new_variable_name):
        print(variable_name, "  --->  ", new_variable_name)
        param = reader.get_tensor(variable_name)
        return param

    def train_graph(self):

        self.masked_position = tf.placeholder('int32',
                                              [None, None], name='masked_position')
        self.itemseq_output = tf.placeholder('int32',
                                             [None, None], name='itemseq_output')
        self.masked_items = tf.placeholder('int32',
                                           [None, None], name='masked_items')
        self.label_weights = tf.placeholder(tf.float32,
                                            [None, None], name='label_weights')

        context_seq = self.itemseq_input
        label_seq = self.label_weights

        self.dilate_input = self.model_graph(context_seq, train=True)

        self.loss = self.get_masked_lm_output(self.model_para, self.dilate_input,
                                              self.masked_position,
                                              self.masked_items, label_seq, trainable=True)

    def model_graph(self, itemseq_input, train=True):
        model_para = self.model_para

        self.context_embedding = tf.nn.embedding_lookup(self.allitem_embeddings,
                                                        itemseq_input, name="context_embedding")
        dilate_input = self.context_embedding
        layer_num = len(model_para['dilations'])


        for layer_id, dilation in enumerate(model_para['dilations']):
            if self.load_model:
                dilate_input = ops.nextitnet_residual_block_alpha(dilate_input, dilation,
                                                                   layer_id, self.method,
                                                                   model_para['dilated_channels'],
                                                                   model_para['kernel_size'], self.reader, layer_num,
                                                                   train=train)

            else:
                dilate_input = ops.nextitnet_residual_block_alpha(dilate_input, dilation,
                                                                layer_id, self.method, model_para['dilated_channels'],
                                                                model_para['kernel_size'], None, layer_num, train=train)


        return dilate_input

    def predict_graph(self, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        context_seq = self.itemseq_input

        dilate_input = self.model_graph(context_seq, train=False)
        model_para = self.model_para

        logits_2D = tf.reshape(dilate_input[:, -1:, :], [-1, model_para['dilated_channels']])
        logits_2D = tf.matmul(logits_2D, tf.transpose(self.softmax_w))
        logits_2D = tf.nn.bias_add(logits_2D, self.softmax_b)


        probs_flat = tf.nn.softmax(logits_2D)

        self.g_probs = tf.reshape(probs_flat, [-1, 1, model_para['item_size']])
        self.top_10 = tf.nn.top_k(self.g_probs, 10)  # top10
        self.top_5 = tf.nn.top_k(self.g_probs, 5)

    def gather_indexes(self, sequence_tensor, positions):
        """Gathers the vectors at the specific positions over a minibatch."""
        sequence_shape = utils_GRec.get_shape_list(sequence_tensor, expected_rank=3)
        batch_size = sequence_shape[0]
        seq_length = sequence_shape[1]
        width = sequence_shape[2]

        flat_offsets = tf.reshape(
            tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
        flat_positions = tf.reshape(positions + flat_offsets, [-1])
        flat_sequence_tensor = tf.reshape(sequence_tensor,
                                          [batch_size * seq_length, width])
        output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
        return output_tensor

    def get_masked_lm_output(self, bert_config, input_tensor, positions,
                             label_ids, label_weights, trainable=True):
        """Get loss and log probs for the masked LM."""

        input_tensor = self.gather_indexes(input_tensor, positions)

        logits_2D = input_tensor
        label_flat = tf.reshape(label_ids, [-1, 1])  # 1 is the number of positive example
        num_sampled = int(0.2 * self.model_para['item_size'])  # sample 20% as negatives
        loss = tf.nn.sampled_softmax_loss(self.softmax_w, self.softmax_b, label_flat, logits_2D,
                                          num_sampled,
                                          self.model_para['item_size'])

        loss = tf.reduce_mean(loss)
        regularization = self.L2 * tf.reduce_mean([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        loss = loss + regularization

        return loss





















