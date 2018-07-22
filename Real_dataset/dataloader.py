import numpy as np


class Gen_Data_loader():
    def __init__(self, batch_size, vocab_dict):
        self.batch_size = batch_size
        self.token_stream = []
        self.vocab_size = 0
        self.vocab_dict = vocab_dict

    def create_batches(self, data_file_list):
        """make self.token_stream into a integer stream."""
        self.token_stream = []
        print("load %s file data.." % ' '.join(data_file_list))
        for data_file in data_file_list:
            with open(data_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    line = line.split()
                    parse_line = [int(x) for x in line]
                    self.token_stream.append(parse_line)

        self.num_batch = int(len(self.token_stream) / self.batch_size)
        # cut the taken_stream's length exactly equal to num_batch * batch_size
        self.token_stream = self.token_stream[:self.num_batch * self.batch_size]
        self.sequence_batch = np.split(np.array(self.token_stream), self.num_batch, 0)
        self.pointer = 0
        print("      Load %d * %d batches" % (self.num_batch, self.batch_size))

    def next_batch(self):
        """take next batch by self.pointer"""
        ret = self.sequence_batch[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0


class Dis_Data_loader():
    def __init__(self, batch_size, vocab_dict, max_sequence_length):
        self.batch_size = batch_size
        self.sentences = np.array([])
        self.labels = np.array([])
        self.vocab_dict = vocab_dict
        self.max_sequence_length = max_sequence_length

    def load_train_data(self, positive_file_list, negative_file_list):
        # Load data
        positive_examples = []
        negative_examples = []
        for positive_file in positive_file_list:
            with open(positive_file)as fin:
                for line in fin:
                    line = line.strip()
                    line = line.split()
                    parse_line = [int(x) for x in line]
                    positive_examples.append(parse_line)
        for negative_file in negative_file_list:
            with open(negative_file)as fin:
                for line in fin:
                    line = line.strip()
                    line = line.split()
                    parse_line = [int(x) for x in line]
                    negative_examples.append(parse_line)

        self.sentences = np.array(positive_examples + negative_examples)
        self.sentences = self.padding(self.sentences, self.max_sequence_length)

        # Generate labels
        positive_labels = [[0, 1] for _ in positive_examples]
        negative_labels = [[1, 0] for _ in negative_examples]
        self.labels = np.concatenate([positive_labels, negative_labels], 0)

        # Shuffle the data
        shuffle_indices = np.random.permutation(np.arange(len(self.labels)))
        self.sentences = self.sentences[shuffle_indices]
        self.labels = self.labels[shuffle_indices]

        # Split batches
        self.num_batch = int(len(self.labels) / self.batch_size)
        self.sentences = self.sentences[:self.num_batch * self.batch_size]
        self.labels = self.labels[:self.num_batch * self.batch_size]
        self.sentences_batches = np.split(self.sentences, self.num_batch, 0)
        self.labels_batches = np.split(self.labels, self.num_batch, 0)
        self.pointer = 0

    def next_batch(self):
        """take next batch (sentence, label) by self.pointer"""
        ret = self.sentences_batches[self.pointer], self.labels_batches[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0

    def padding(self, inputs, max_sequence_length):
        batch_size = len(inputs)
        inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32)  # == PAD
        for i, seq in enumerate(inputs):
            for j, element in enumerate(seq):
                inputs_batch_major[i, j] = element
        return inputs_batch_major

