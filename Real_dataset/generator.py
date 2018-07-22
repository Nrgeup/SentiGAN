import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops, control_flow_ops
from tensorflow.python.layers import core as layers_core
import numpy as np


class Generator(object):
    def __init__(self, num_emb, vocab_dict, batch_size, emb_dim, num_units,
                 max_sequence_length, learning_rate=0.01, reward_gamma=0.95,
                 ):
        self.num_emb = num_emb
        self.vocab_dict = vocab_dict
        self.batch_size = batch_size
        self.emb_dim = emb_dim
        self.num_units = num_units
        self.max_sequence_length = max_sequence_length
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.reward_gamma = reward_gamma
        self.grad_clip = 5.0
        self.keep_prob = 1.0

        self.g_embeddings = tf.Variable(self.init_matrix([self.num_emb, self.emb_dim]))

        with tf.variable_scope('placeholder'):
            self.x = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_sequence_length])
            self.sequence_lengths = tf.placeholder(tf.int32, shape=[self.batch_size])

        with tf.variable_scope('embedding'):
            # batch_size major
            self.emb_x = tf.nn.embedding_lookup(self.g_embeddings, self.x)

        with tf.variable_scope('projection'):
            self.output_layer = layers_core.Dense(self.num_emb, use_bias=False)

        def _get_cell(_num_units):
            return tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(_num_units),
                                                 input_keep_prob=self.keep_prob)

        with tf.variable_scope("decoder"):
            self.decoder_cell = _get_cell(self.num_units)

            # inital_states
            self.c = tf.random_normal([self.batch_size, self.num_units], mean=0, stddev=4)
            self.h = tf.random_normal([self.batch_size, self.num_units], mean=0, stddev=4)

            # self.c = tf.placeholder(tf.float32, shape=[self.batch_size, self.num_units])
            # self.h = tf.placeholder(tf.float32, shape=[self.batch_size, self.num_units])

            # tf.zeros([self.batch_size, self.num_units])
            # h = tf.zeros([self.batch_size, self.num_units])
            self.initial_state = tf.contrib.rnn.LSTMStateTuple(c=self.c, h=self.h)

            ###################### pretain with targets ######################
            helper_pt = tf.contrib.seq2seq.TrainingHelper(
                inputs=self.emb_x,
                sequence_length=self.sequence_lengths,
                time_major=False,
            )
            decoder_pt = tf.contrib.seq2seq.BasicDecoder(
                cell=self.decoder_cell,
                helper=helper_pt,
                initial_state=self.initial_state,
                output_layer=self.output_layer
            )

            outputs_pt, _final_state, sequence_lengths_pt = tf.contrib.seq2seq.dynamic_decode(
                decoder=decoder_pt,
                output_time_major=False,
                maximum_iterations=self.max_sequence_length,
                swap_memory=True,
            )
            self.logits_pt = outputs_pt.rnn_output

            self.g_predictions = tf.nn.softmax(self.logits_pt)

            self.targets = tf.placeholder(dtype=tf.int32, shape=[None, None])
            self.target_weights = tf.placeholder(dtype=tf.float32, shape=[None, None])

            crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.targets, logits=self.logits_pt)
            self.pretrain_loss = tf.reduce_sum(crossent * self.target_weights) / tf.to_float(self.batch_size)

            self.global_step = tf.Variable(0, trainable=False)
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            gradients, v = zip(*optimizer.compute_gradients(self.pretrain_loss))
            gradients, _ = tf.clip_by_global_norm(gradients, self.grad_clip)
            self.pretrain_updates = optimizer.apply_gradients(zip(gradients, v), global_step=self.global_step)

            ################## gan loss with rewards  #####################
            self.rewards = tf.placeholder(dtype=tf.float32, shape=[None, None])
            self.rewards_loss = tf.reduce_sum(
                tf.reduce_sum(
                    tf.one_hot(tf.to_int32(tf.reshape(self.x, [-1])), self.num_emb, 1.0, 0.0) * tf.clip_by_value(
                        tf.reshape(self.g_predictions, [-1, self.num_emb]), 1e-20, 1.0)
                    , 1) * tf.reshape(self.rewards, [-1])  # * tf.reshape(self.target_weights, [-1])
            )
            optimizer_gan = tf.train.RMSPropOptimizer(self.learning_rate)
            gradients_gan, v_gan = zip(*optimizer_gan.compute_gradients(self.rewards_loss))
            gradients_gan, _gan = tf.clip_by_global_norm(gradients_gan, self.grad_clip)
            self.rewards_updates = optimizer_gan.apply_gradients(zip(gradients_gan, v_gan), global_step=self.global_step)


            ###################### train without targets ######################
            helper_o = tf.contrib.seq2seq.SampleEmbeddingHelper(
                self.g_embeddings,
                tf.fill([self.batch_size], self.vocab_dict['<GO>']),
                end_token=self.vocab_dict['<EOS>']
            )
            decoder_o = tf.contrib.seq2seq.BasicDecoder(
                cell=self.decoder_cell,
                helper=helper_o,
                initial_state=self.initial_state,
                output_layer=self.output_layer
            )
            outputs_o, _final_state_o, sequence_lengths_o = tf.contrib.seq2seq.dynamic_decode(
                decoder=decoder_o,
                output_time_major=False,
                maximum_iterations=self.max_sequence_length,
                swap_memory=True,
            )

            self.out_lenghts = sequence_lengths_o
            self.out_tokens = tf.unstack(outputs_o.sample_id, axis=0)

            ######################  infer  ######################
            helper_i = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                self.g_embeddings,
                tf.fill([self.batch_size], self.vocab_dict['<GO>']),
                end_token=self.vocab_dict['<EOS>']
            )
            decoder_i = tf.contrib.seq2seq.BasicDecoder(
                cell=self.decoder_cell,
                helper=helper_i,
                initial_state=self.initial_state,
                output_layer=self.output_layer
            )
            
            # beam_search_initial_state = tf.contrib.seq2seq.tile_batch(
            #     self.initial_state,
            #     multiplier=20)
            # decoder_i = tf.contrib.seq2seq.BeamSearchDecoder(
            #     cell=self.decoder_cell,
            #     embedding=self.g_embeddings,
            #     start_tokens=tf.fill([self.batch_size], self.vocab_dict['<GO>']),
            #     end_token=self.vocab_dict['<EOS>'],
            #     initial_state=beam_search_initial_state,
            #     beam_width=20,
            #     output_layer=self.output_layer,
            #     length_penalty_weight=0.1,
            # )

            outputs_i, _final_state_i, sequence_lengths_i = tf.contrib.seq2seq.dynamic_decode(
                decoder=decoder_i,
                output_time_major=False,
                maximum_iterations=self.max_sequence_length,
                swap_memory=True,
            )

            # only for beam search
            # sample_id = tf.transpose(outputs_i.predicted_ids, perm=[0,2,1])

            sample_id = outputs_i.sample_id

            self.infer_tokens = tf.unstack(sample_id, axis=0)


            ###################### rollout ######################
            self.rollout_input_ids = tf.placeholder(dtype=tf.int32, shape=[None, None])
            self.rollout_input_length = tf.placeholder(dtype=tf.int32, shape=())
            self.rollout_input_lengths = tf.placeholder(dtype=tf.int32, shape=[None])
            self.rollout_next_id = tf.placeholder(dtype=tf.int32, shape=[None])

            rollout_inputs = tf.nn.embedding_lookup(self.g_embeddings, self.rollout_input_ids)
            helper_ro = tf.contrib.seq2seq.TrainingHelper(
                rollout_inputs,
                self.rollout_input_lengths
            )
            rollout_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=self.decoder_cell,
                helper=helper_ro,
                initial_state=self.initial_state,
                output_layer=self.output_layer
            )
            _, final_state_ro, _ = tf.contrib.seq2seq.dynamic_decode(
                rollout_decoder,
                maximum_iterations=self.max_sequence_length,
                swap_memory=True
            )
            initial_state_MC = final_state_ro
            helper_MC = tf.contrib.seq2seq.SampleEmbeddingHelper(
                self.g_embeddings,
                self.rollout_next_id,
                end_token=self.vocab_dict['<EOS>']
            )
            rollout_decoder_MC = tf.contrib.seq2seq.BasicDecoder(
                cell=self.decoder_cell,
                helper=helper_MC,
                initial_state=initial_state_MC,
                output_layer=self.output_layer
            )
            self.max_mc_length = tf.cast(self.max_sequence_length - self.rollout_input_length, tf.int32)
            decoder_output_MC, _, _ = tf.contrib.seq2seq.dynamic_decode(
                rollout_decoder_MC,
                output_time_major=False,
                maximum_iterations=self.max_mc_length,
                swap_memory=True
            )
            self.sample_id_MC = decoder_output_MC.sample_id

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

    def pretrain_step(self, sess, x):
        input_x, lengths_x = self.pad_input_data(x)
        target_x = self.pad_target_data(x)
        target_weights = self.get_weights(lengths_x)

        outputs = sess.run([self.pretrain_updates, self.pretrain_loss], feed_dict={
            self.x: input_x,
            self.sequence_lengths: [self.max_sequence_length] * self.batch_size,
            self.targets: target_x,
            self.target_weights: target_weights
        })
        return outputs

    def update_with_rewards(self, sess, x, rewards):
        input_x, lengths_x = self.pad_input_data(x)
        target_x = self.pad_target_data(x)
        target_weights = self.get_weights(lengths_x)

        [rewards_updates, rewards_loss] = sess.run([self.rewards_updates, self.rewards_loss], feed_dict={
            self.x: input_x,
            self.sequence_lengths: [self.max_sequence_length] * self.batch_size,
            self.targets: target_x,
            self.rewards: rewards,
            self.target_weights: target_weights
        })
        return rewards_loss

    def generate(self, sess):
        [outputs] = sess.run([self.out_tokens])
        # outputs_new = self.delete_output_data(outputs, out_lenghts)
        return outputs  # , out_lenghts

    def infer(self, sess):
        [outputs] = sess.run([self.infer_tokens])
        # outputs_new = self.delete_output_data(outputs, out_lenghts)
        return outputs

    def init_matrix(self, shape):
        return tf.random_normal(shape, stddev=0.1)

    def _get_cell(self, num_units):
        return tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(num_units),
                                             input_keep_prob=self.keep_prob)

    def pad_input_data(self, x):
        max_l = self.max_sequence_length
        go_id = self.vocab_dict['<GO>']
        end_id = self.vocab_dict['<EOS>']
        x_len = len(x)
        ans = np.zeros((x_len, max_l), dtype=int)
        ans_lengths = []
        for i in range(x_len):
            ans[i][0] = go_id
            jj = min(len(x[i]), self.max_sequence_length - 2)
            for j in range(jj):
                ans[i][j + 1] = x[i][j]
            ans[i][jj+1] = end_id
            ans_lengths.append(jj + 2)
        return ans, ans_lengths

    def pad_target_data(self, x):
        max_l = self.max_sequence_length
        end_id = self.vocab_dict['<EOS>']
        x_len = len(x)
        ans = np.zeros((x_len, max_l), dtype=int)
        for i in range(x_len):
            jj = min(len(x[i]), max_l-1)
            for j in range(jj):
                ans[i][j] = x[i][j]
            ans[i][jj] = end_id
        return ans

    def delete_output_data(self, x, lengths):
        ans = []
        for i, item in enumerate(x):
            ans.append(item[:lengths[i]])
        return np.array(ans)

    def get_weights(self, lengths):
        x_len = len(lengths)
        max_l = self.max_sequence_length
        ans = np.zeros((x_len, max_l))
        for ll in range(x_len):
            kk = lengths[ll] - 1
            for jj in range(kk):
                ans[ll][jj] = 1/float(kk)
        return ans

    def get_reward(self, sess, input_x, rollout_num, discriminator):
        # x = self.pad_input_data(input_x, go_id)
        x, lengths_x = self.pad_input_data(input_x)
        input_x = self.padding(input_x, self.max_sequence_length)
        rewards = []

        for i in range(rollout_num):
            for given_num in range(1, self.max_sequence_length):

                rollout_next_id = []
                for _item in x:
                    rollout_next_id.append(_item[given_num])

                feed = {
                    self.rollout_input_ids: x,
                    self.rollout_input_length: given_num,
                    self.rollout_input_lengths: [given_num] * self.batch_size,
                    self.rollout_next_id: rollout_next_id
                }

                # print(x[0])
                # print("given_num:%d" % given_num)
                # print("roll_next_id[0]:\n%s" % str(rollout_next_id[0]))

                mc_samples = sess.run(self.sample_id_MC, feed)

                # print("mc_samples[0]:")
                # print(mc_samples[0])
                #
                fix_samples = np.array(input_x)[:, 0: given_num]
                # print(fix_samples[0])

                samples = np.concatenate((fix_samples, mc_samples), axis=1)
                # print(samples[0])
                # input("sad!!!!!")
                # print("samples[0]")
                # print(samples[0])
                samples = self.padding(samples, self.max_sequence_length)
                # print("samples[0]")
                # print(samples[0])

                feed = {discriminator.input_x: samples, discriminator.dropout_keep_prob: 1.0}
                ypred_for_auc = sess.run(discriminator.ypred_for_auc, feed)
                ypred = np.array([item[0] for item in ypred_for_auc])
                if i == 0:
                    rewards.append(ypred)
                else:
                    rewards[given_num - 1] += ypred

            # the last token reward
            feed = {discriminator.input_x: input_x, discriminator.dropout_keep_prob: 1.0}
            ypred_for_auc = sess.run(discriminator.ypred_for_auc, feed)
            ypred = np.array([item[0] for item in ypred_for_auc])
            if i == 0:
                rewards.append(ypred)
            else:
                rewards[self.max_sequence_length - 1] += ypred
            # print(len(rewards))
            # print(len(rewards[0]))
            # print(rewards[0])
            # input()

        rewards = np.transpose(np.array(rewards)) / (1.0 * rollout_num)  # batch_size x seq_length
        rewards = self.get_new_rewards(lengths_x, rewards)
        return rewards

    def get_new_rewards(self, lengths_x, rewards):
        r = len(rewards[0])
        for i in range(len(lengths_x)):
            l = lengths_x[i]
            for j in range(l, r):
                rewards[i][j] = rewards[i][-1]
        return rewards



    def padding(self, inputs, max_sequence_length):
        batch_size = len(inputs)
        inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32)  # == PAD
        for i, seq in enumerate(inputs):
            for j, element in enumerate(seq):
                inputs_batch_major[i, j] = element
        return inputs_batch_major

    def save_model(self, sess):
        self.saver.save(sess, 'save/ckpt/model.ckpt')
        print("save model success!")

