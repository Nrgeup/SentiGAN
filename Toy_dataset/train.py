import numpy as np
import tensorflow as tf
import random
from dataloader import Gen_Data_loader, Dis_Data_loader
import pickle
from target_lstm import TARGET_LSTM
from generator import Generator
from discriminator import Discriminator
# from rollout import ROLLOUT


#########################################################################################
#  Generator  Hyper-parameters
######################################################################################
EMB_DIM = 32 # embedding dimension
HIDDEN_DIM = 32 # hidden state dimension of lstm cell
SEQ_LENGTH = 20 # sequence length

SEED = 88
BATCH_SIZE = 64


#########################################################################################
#  Discriminator  Hyper-parameters
#########################################################################################
dis_embedding_dim = 64
dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
dis_dropout_keep_prob = 0.75
dis_l2_reg_lambda = 0.2
dis_batch_size = 64


#########################################################################################
#  Basic Training Parameters
#########################################################################################
TOTAL_BATCH = 200
positive_file = 'save/real_data.txt'
negative_file = 'save/generator_sample.txt'
eval_file = 'save/eval_file.txt'
generated_num = 10000
vocab_size = 5000
START_TOKEN = 0


def generate_samples_from_target(sess, trainable_model, batch_size, generated_num, output_file):
    # Generate Target Samples
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        generated_samples.extend(trainable_model.generate(sess))

    with open(output_file, 'w') as fout:
        for poem in generated_samples:
            buffer = ' '.join([str(x) for x in poem]) + '\n'
            fout.write(buffer)


def generate_samples(sess, trainable_model, batch_size, generated_num, output_file):
    # Generate Samples
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        generated_samples.extend(trainable_model.generate(sess))

    with open(output_file, 'w') as fout:
        for poem in generated_samples:
            buffer = ' '.join([str(x) for x in poem]) + '\n'
            fout.write(buffer)


def target_loss(sess, target_lstm, data_loader):
    # target_loss means the oracle negative log-likelihood tested with the oracle model "target_lstm"
    # For more details, please see the Section 4 in https://arxiv.org/abs/1609.05473
    nll = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        batch = data_loader.next_batch()
        g_loss = sess.run(target_lstm.pretrain_loss, {target_lstm.x: batch})
        nll.append(g_loss)

    return np.mean(nll)


def pre_train_epoch(sess, trainable_model, data_loader):
    # Pre-train the generator using MLE for one epoch
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        batch = data_loader.next_batch()
        _, g_loss = trainable_model.pretrain_step(sess, batch, START_TOKEN)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)


# def train_with_good_rewards(sess, data_loader):
#     data_loader.reset_pointer()



def main():
    random.seed(SEED)
    np.random.seed(SEED)


    # prepare data
    gen_data_loader = Gen_Data_loader(BATCH_SIZE)
    likelihood_data_loader = Gen_Data_loader(BATCH_SIZE)  # For testing
    dis_data_loader = Dis_Data_loader(BATCH_SIZE)


    generator = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN)


    # target_params's size: [15 * 5000 * 32]
    target_params = pickle.load(open('./save/target_params_py3.pkl', 'rb'))
    # The oracle model
    target_lstm = TARGET_LSTM(5000, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, 20, 0, target_params)

    discriminator = Discriminator(sequence_length=20, num_classes=2, vocab_size=vocab_size,
                                  embedding_size=dis_embedding_dim,
                                  filter_sizes=dis_filter_sizes, num_filters=dis_num_filters,
                                  l2_reg_lambda=dis_l2_reg_lambda)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    generate_samples_from_target(sess, target_lstm, BATCH_SIZE, generated_num, positive_file)
    gen_data_loader.create_batches(positive_file)

    # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    #
    # likelihood_data_loader.create_batches(positive_file)
    # for i in range(100):
    #     test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
    #     print('my step ', i, 'test_loss ', test_loss)
    #     input("next:")
    # input("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    log = open('save/experiment-log.txt', 'w')
    #  pre-train generator
    print('Start pre-training...')
    log.write('pre-training...\n')
    ans_file = open("learning_cure.txt", 'w')
    for epoch in range(120):  # 120
        loss = pre_train_epoch(sess, generator, gen_data_loader)
        if epoch % 1 == 0:
            generate_samples(sess, generator, BATCH_SIZE, generated_num, eval_file)
            likelihood_data_loader.create_batches(eval_file)
            test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
            print('pre-train epoch ', epoch, 'test_loss ', test_loss)
            buffer = 'epoch:\t' + str(epoch) + '\tnll:\t' + str(test_loss) + '\n'
            log.write(buffer)
            ans_file.write("%s\n" % str(test_loss))

    buffer = 'Start pre-training discriminator...'
    print(buffer)
    log.write(buffer)
    for _ in range(10):   # 10
        generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)
        dis_data_loader.load_train_data(positive_file, negative_file)
        for _ in range(3):
            dis_data_loader.reset_pointer()
            for it in range(dis_data_loader.num_batch):
                x_batch, y_batch = dis_data_loader.next_batch()
                feed = {
                    discriminator.input_x: x_batch,
                    discriminator.input_y: y_batch,
                    discriminator.dropout_keep_prob: dis_dropout_keep_prob,
                }
                d_loss, d_acc, _ = sess.run([discriminator.loss, discriminator.accuracy, discriminator.train_op], feed)
        buffer = "discriminator loss %f acc %f\n" % (d_loss, d_acc)
        print(buffer)

        log.write(buffer)
    ans_file.write("==========\n")
    print("Start Adversarial Training...")
    log.write('adversarial training...')
    for total_batch in range(TOTAL_BATCH):
        # Train the generator
        for it in range(1):
            samples = generator.generate(sess)
            rewards = generator.get_reward(sess, samples, 16, discriminator, START_TOKEN)
            a = str(samples[0])
            b = str(rewards[0])
            buffer = "%s\n%s\n\n" % (a, b)
            # print(buffer)
            log.write(buffer)
            rewards_loss = generator.update_with_rewards(sess, samples, rewards, START_TOKEN)

            # good rewards
            # good_samples = gen_data_loader.next_batch()
            # rewards = np.array([[1.0] * SEQ_LENGTH] * BATCH_SIZE)
            # a = str(good_samples[0])
            # b = str(rewards[0])
            # buffer = "%s\n%s\n\n" % (a, b)
            # print(buffer)
            # log.write(buffer)
            # rewards_loss = generator.update_with_rewards(sess, good_samples, rewards, START_TOKEN)

            # little1 good reward
            # litter1_samples = gen_data_loader.next_batch()
            # rewards = generator.get_reward(sess, litter1_samples, 16, discriminator, START_TOKEN)
            # a = str(little1 good reward[0])
            # b = str(rewards[0])
            # buffer = "%s\n%s\n\n" % (a, b)
            # print(buffer)
            # log.write(buffer)
            # rewards_loss = generator.update_with_rewards(sess, litter1_samples, rewards, START_TOKEN)


        # Test
        if total_batch % 1 == 0 or total_batch == TOTAL_BATCH - 1:
            generate_samples(sess, generator, BATCH_SIZE, generated_num, eval_file)
            likelihood_data_loader.create_batches(eval_file)
            test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
            buffer = 'reward-train epoch %s train loss %s test_loss %s\n' % (str(total_batch), str(rewards_loss), str(test_loss))
            print(buffer)
            log.write(buffer)
            ans_file.write("%s\n" % str(test_loss))

        # Train the discriminator
        for _ in range(1):
            generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)
            dis_data_loader.load_train_data(positive_file, negative_file)
            for _ in range(3):
                dis_data_loader.reset_pointer()
                for it in range(dis_data_loader.num_batch):
                    x_batch, y_batch = dis_data_loader.next_batch()
                    feed = {
                        discriminator.input_x: x_batch,
                        discriminator.input_y: y_batch,
                        discriminator.dropout_keep_prob: dis_dropout_keep_prob,
                    }
                    d_loss, d_acc, _ = sess.run([discriminator.loss, discriminator.accuracy, discriminator.train_op], feed)
            if total_batch % 5 == 0 or total_batch == TOTAL_BATCH - 1:
                buffer = "discriminator loss %f acc %f\n" % (d_loss, d_acc)
                print(buffer)
                log.write(buffer)


if __name__ == '__main__':
    main()



