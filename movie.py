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

        self.q_clips = pickle.load(open('./movieqa/q_clips.p', 'rb'))
        # change this to your audio base
        self.audio_base = '/home/shijie/Downloads/features/features/melspectrogram_128/all_video_clips'
        self.postfix = '.orig.spec.npy'

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

     
        for start_idx in range(0, len(self.qids), self.batch_size):
            end_idx = min(start_idx + self.batch_size, len(self.qids))
            batch_question = []
            batch_audio = []
            batch_subtitles = []
            batch_answers = []
            batch_correct_index = []
            for idx in range(start_idx, end_idx):
                order_idx = order[idx]
                question, subtitles, answers, correct_idx = self.__getitem__(order_idx)
                batch_question.append(question)
                batch_subtitles.append(subtitles)
                batch_answers.append(answers)
                batch_correct_index.append(correct_idx)

                video_names = self.q_clips[self.qids[order_idx]]
                ### for debugging purpose ###
                video_names = random.sample(['tt0086879.sf-211630.ef-217006.video.mp4', 'tt0125439.sf-016072.ef-016235.video.mp4', 'tt0373889.sf-178244.ef-178933.video.mp4', 'tt1270798.sf-066306.ef-069733.video.mp4', 'tt0412019.sf-118504.ef-121216.video.mp4', 'tt1499658.sf-008895.ef-011364.video.mp4'], random.choice([1, 2, 3, 4]))
                audio = []
                for name in video_names:
                    af = np.load("{}/{}{}".format(self.audio_base, name, self.postfix))
                    af = af[:, ::40]
                    audio.append(af)
                audio1 = np.concatenate(audio, axis=1).tolist()
                batch_audio.append(audio1)

            # ( seq_len, batch_size )
            tensor_question = pad_longest(batch_question)
            tensor_subtitles = pad_longest(batch_subtitles)
            
            list_tensor_answer = [ pad_longest(a) for a in batch_answers ]
            tensor_correct_index = torch.LongTensor(batch_correct_index)

            tensor_audio = torch.stack([pad_longest(list(v)) for v in zip(*batch_audio)]).permute(1, 0, 2)

            yield tensor_question, tensor_audio, tensor_subtitles, list_tensor_answer, tensor_correct_index

if __name__ == "__main__":
    pass
