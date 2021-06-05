import os
from os import listdir
from os.path import isfile, join
import numpy as np
import time


def INFO_LOG(info):
    print("[%s]%s" % (time.strftime("%Y-%m-%d %X", time.localtime()), info))


class Data_Loader:
    def __init__(self, options):
        self.pad = "<PAD>"
        positive_data_file = options['dir_name']
        positive_examples = list(open(positive_data_file, "r").readlines())
        positive_examples = [s for s in positive_examples]

        self.max_document_length = max([len(x.split(",")[1:]) for x in positive_examples])

        users = [int(x.split(",")[0]) for x in positive_examples]
        users = np.reshape(users, (-1, 1))
        self.user_size = len(np.unique(users))
        self.item_fre = {self.pad: 0}

        for sample in positive_examples:
            for item in sample.strip().split(",")[1:]:
                if item in self.item_fre.keys():
                    self.item_fre[item] += 1
                else:
                    self.item_fre[item] = 1
            self.item_fre[self.pad] += self.max_document_length - len(sample.strip().split(",")[1:])

        # count_pairs = sorted(self.item_fre.items(), key=lambda x: (-x[1], x[0]))
        count_pairs = self.item_fre.items()
        self.items_voc, _ = list(zip(*count_pairs))
        self.item2id = dict(zip(self.items_voc, range(len(self.items_voc))))
        self.padid = self.item2id[self.pad]
        self.id2item = {value: key for key, value in self.item2id.items()}

        INFO_LOG("Vocab size:{}".format(self.size()))

        self.items = np.array(self.getSamplesid(positive_examples))
        self.items = np.concatenate((users, self.items), axis=1)

    def sample2id(self, sample):
        sample2id = []
        for s in sample.strip().split(',')[1:]:
            sample2id.append(self.item2id[s])

        sample2id = ([self.padid] * (self.max_document_length - len(sample2id))) + sample2id
        return sample2id

    def getSamplesid(self, samples):
        samples2id = []
        for sample in samples:
            samples2id.append(self.sample2id(sample))

        return samples2id

    def size(self):
        return len(self.item2id)



