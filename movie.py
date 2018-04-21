import itertools
import pickle
import random

import numpy as np
import torch
import torch.utils.data as data

import config

def pad_longest(v, fillvalue=0):
    arr = np.array(list(itertools.zip_longest(*v, fillvalue=fillvalue)))
    arr = torch.LongTensor(arr)
    return arr

def get_dataset(train=False, val=False, test=False):
    """ Returns a data loader for the desired split """
    assert train + val + test == 1, 'need to set exactly one of {train, val, test} to True'
    if train:
        split = 'train'
    elif val:
        split = 'val'
    else:
        split = 'test'
    data_pickle = './movieqa/movieqa.{}.pickle'.format(split)
    vocab_pickle = './movieqa/movieqa.vocab'
    dataset = MovieQADataset(data_pickle, vocab_pickle, config.batch_size, shuffle=train)
    return dataset

class MovieQADataset(object):
    def __init__(self, data_pickle, vocab_pickle, batch_size, shuffle=False):
        super(MovieQADataset, self).__init__()
        with open(data_pickle,'rb') as fin:
            data = pickle.load(fin)

        with open(vocab_pickle,'rb') as fin:
            vocab = pickle.load(fin)
   
        self.qids = sorted(data.keys())
        self.data = data

        self.batch_size = batch_size
        self.shuffle = shuffle

        self.vocab = vocab

    def __len__(self):
        return len(self.qids)

    def __getitem__(self, idx):
        qid = self.qids[idx]
        d = self.data[qid]
        q = d['question']
        s = d['subtitles']
        a = d['answers']
        c = d['correct_index']
        return q, s, a, c

    def loader(self):
        order = list(range(len(self.qids)))
        if self.shuffle:
            random.shuffle(order)

        batch_question = []
        batch_subtitles = []
        batch_answers = []
        batch_correct_index = []
     
        for start_idx in range(0, len(self.qids), self.batch_size):
            end_idx = min(start_idx + self.batch_size, len(self.qids))
            for idx in range(start_idx, end_idx):
                order_idx = order[idx]
                question, subtitles, answers, correct_idx = self.__getitem__(order_idx)
                batch_question.append(question)
                batch_subtitles.append(subtitles)
                batch_answers.append(answers)
                batch_correct_index.append(correct_idx)

            # ( seq_len, batch_size )
            tensor_question = pad_longest(batch_question)
            tensor_subtitles = pad_longest(batch_subtitles)
            list_tensor_answer = [ pad_longest(a) for a in batch_answers ]
            tensor_correct_index = torch.LongTensor(batch_correct_index)

            yield tensor_question, tensor_subtitles, list_tensor_answer, tensor_correct_index

if __name__ == "__main__":
    pass