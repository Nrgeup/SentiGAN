import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops, control_flow_ops
from tensorflow.python.layers import core as layers_core
import numpy as np


class Generator(object):
    def __init__(self, num_emb, batch_size, emb_dim, num_units,
                 sequence_length, start_token,
                 learning_rate=0.01, reward_gamma=0.95,
                 ):
        self.num_emb = num_emb
        self.batch_size = batch_size
        self.emb_dim = emb_dim
        self.num_units = num_units
        self.sequence_length = sequence_length
        self.start_token = start_token
        # self.end_token = end_token
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.reward_gamma = reward_gamma
        self.grad_clip = 5.0

        self.sequence_lengths = [self.sequence_length] * self.batch_size
        self.keep_prob = 1.0
        self.max_output_length = self.sequence_length

        self.g_embeddings = tf.Variable(self.init_matrix([self.num_emb, self.emb_dim]))

        with tf.variable_scope('placeholder'):
            self.x = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_output_length])
            self.rewards = tf.placeholder(tf.float32, shape=[self.batch_size, self.sequence_length])

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
            c = tf.zeros([self.batch_size, self.num_units])
            h = tf.zeros([self.batch_size, self.num_units])
            self.initial_state = tf.contrib.rnn.LSTMStateTuple(c=c, h=h)


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
                maximum_iterations=self.max_output_length,
                swap_memory=True,
            )
            self.logits_pt = outputs_pt.rnn_output
            self.targets = tf.placeholder(dtype=tf.int32, shape=[None, None])

            self.g_predictions = tf.nn.softmax(self.logits_pt)

            # self.pretrain_token_id = tf.cast(tf.argmax(self.g_predictions, axis=-1), tf.int32)

            self.pretrain_loss = - tf.reduce_sum(
                tf.one_hot(tf.to_int32(tf.reshape(self.targets, [-1])), self.num_emb, 1.0, 0.0) * tf.log(
                    tf.clip_by_value(tf.reshape(self.g_predictions, [-1, self.num_emb]), 1e-20, 1.0)
                )
            ) / (self.sequence_length * self.batch_size)

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
                    , 1) * tf.reshape(self.rewards, [-1])
            )
            optimizer_gan = tf.train.RMSPropOptimizer(self.learning_rate)
            gradients_gan, v_gan = zip(*optimizer_gan.compute_gradients(self.rewards_loss))
            gradients_gan, _gan = tf.clip_by_global_norm(gradients_gan, self.grad_clip)
            self.rewards_updates = optimizer_gan.apply_gradients(zip(gradients_gan, v_gan), global_step=self.global_step)


            ###################### train without targets ######################
            helper_o = tf.contrib.seq2seq.SampleEmbeddingHelper(
                self.g_embeddings,
                tf.fill([self.batch_size], self.start_token),
                end_token=5001  # self.end_token
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
                maximum_iterations=self.max_output_length,
                swap_memory=True,
            )

            self.out_tokens = tf.unstack(outputs_o.sample_id, axis=0)


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
                maximum_iterations=self.max_output_length,
                swap_memory=True
            )
            initial_state_MC = final_state_ro
            helper_MC = tf.contrib.seq2seq.SampleEmbeddingHelper(
                self.g_embeddings,
                self.rollout_next_id,
                end_token=5001 # self.end_token
            )
            rollout_decoder_MC = tf.contrib.seq2seq.BasicDecoder(
                cell=self.decoder_cell,
                helper=helper_MC,
                initial_state=initial_state_MC,
                output_layer=self.output_layer
            )
            self.max_mc_length = tf.cast(self.max_output_length - self.rollout_input_length, tf.int32)
            decoder_output_MC, _, _ = tf.contrib.seq2seq.dynamic_decode(
                rollout_decoder_MC,
                output_time_major=False,
                maximum_iterations=self.max_mc_length,
                swap_memory=True
            )
            self.sample_id_MC = decoder_output_MC.sample_id



        self.saver = tf.train.Saver(tf.global_variables())

    def pretrain_step(self, sess, x, go_id):
        input_x = self.pad_input_data(x, go_id)
        target_x = x  # self.pad_target_data(x)

        outputs = sess.run([self.pretrain_updates, self.pretrain_loss], feed_dict={
            self.x: input_x,
            self.targets: target_x,
            # self.target_weights: target_weights
        })
        return outputs

    def update_with_rewards(self, sess, x, rewards, go_id):
        input_x = self.pad_input_data(x, go_id)
        target_x = x  # self.pad_target_data(x, end_id)
        # np.array([[1 / float(20)] * 20 + [0.0]] * self.batch_size)

        [rewards_updates, rewards_loss] = sess.run([self.rewards_updates, self.rewards_loss], feed_dict={
            self.x: input_x,
            self.targets: target_x,
            self.rewards: rewards
        })
        return rewards_loss

    def generate(self, sess):
        outputs = sess.run(self.out_tokens)
        # outputs = self.delete_output_data(outputs, self.sequence_length)
        if len(outputs[0]) != 20:
            print("warning! : outputs length:%d" % len(outputs[0]))
            outputs = np.concatenate((outputs, np.array([[5000] * (20 - len(outputs[0]))] * self.batch_size)), axis=1)
        return outputs

    def init_matrix(self, shape):
        return tf.random_normal(shape, stddev=0.1)

    def _get_cell(self, num_units):
        return tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(num_units),
                                             input_keep_prob=self.keep_prob)

    def pad_input_data(self, x, go_id):
        x = np.delete(x, -1, axis=1)
        go_array = np.array([[go_id]] * len(x))
        ans = np.concatenate((go_array, x), axis=1)
        return ans

    # def pad_target_data(self, x, end_id):
    #     end_array = np.array([[end_id]] * len(x))
    #     ans = np.concatenate((x, end_array), axis=1)
    #     return ans

    # def delete_output_data(self, x, length):
    #     ans = []
    #     for item in x:
    #         ans.append(item[:length])
    #     return np.array(ans)

    def get_reward(self, sess, input_x, rollout_num, discriminator, go_id):
        x = self.pad_input_data(input_x, go_id)
        rewards = []

        for i in range(rollout_num):
            for given_num in range(1, 20):

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

                # print("samples[0]:")
                # print(mc_samples[0])

                fix_samples = np.array(input_x)[:, 0: given_num]
                # print(fix_samples[0])
                # b = mc_samples[:, given_num: 20 - given_num]
                # print(b[0])

                samples = np.concatenate((fix_samples, mc_samples), axis=1)
                # print(samples[0])
                # input("mmp,gun!!!!!")
                # continue

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
                rewards[19] += ypred
            # print(len(rewards))
            # print(len(rewards[0]))
            # print(rewards[0])
            # input()

        rewards = np.transpose(np.array(rewards)) / (1.0 * rollout_num)  # batch_size x seq_length
        return rewards

