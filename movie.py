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
        self.audio_base = config.audio_path
        self.audio_postfix = config.audio_postfix
        self.video_base = config.video_path
        self.video_postfix = config.video_postfix

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
        use_audio = True if config.weight_qa > 0 else False
        use_video = True if config.weight_qv > 0 else False
        
        order = list(range(len(self.qids)))
        if self.shuffle:
            random.shuffle(order)

     
        for start_idx in range(0, len(self.qids), self.batch_size):
            end_idx = min(start_idx + self.batch_size, len(self.qids))
            batch_question = []
            batch_audio = []
            batch_images = []
            batch_subtitles = []
            batch_answers = []
            batch_correct_index = []
            batch_indicator = []
            for idx in range(start_idx, end_idx):
                order_idx = order[idx]
                question, subtitles, answers, correct_idx = self.__getitem__(order_idx)
                batch_question.append(question)
                batch_subtitles.append(subtitles)
                batch_indicator.append(self.qids[order_idx])

                if self.shuffle:
                    # Answer shuffling
                    print("Using answer shuffling")
                    tp1 = []
                    for i in range(5):
                        if i == correct_idx:
                            tp1.append(1)
                        else:
                            tp1.append(0)
                        
                    tmp = list(zip(answers, tp1))
                    random.shuffle(tmp)
                    answers = [m[0] for m in tmp]
                
                    for i in range(5):
                        if tmp[i][1] == 1:
                            correct_idx = i
                            break
                
                batch_answers.append(answers)
                batch_correct_index.append(correct_idx)

                video_names = self.q_clips[self.qids[order_idx]]
                if use_audio:
                    audio = []
                    for name in video_names:
                        af = np.load("{}{}{}".format(self.audio_base, name[:name.find('.video')], self.audio_postfix))
                    
                        af = af.T[:, ::40]
                        audio.append(af)
                    audio1 = np.concatenate(audio, axis=1).tolist()
                    batch_audio.append(audio1)
                        
                if use_video:
                    video = []
                    for name in video_names:
                        vf = np.load("{}{}{}".format(self.video_base, name[:name.find('.video')], self.video_postfix))
                        vf = vf.reshape(-1, 512)
                        vf = vf.T[:, ::100]
                        video.append(vf)
                    video1 = np.concatenate(video, axis=1).tolist()
                    batch_images.append(video1)

            # ( seq_len, batch_size )
            tensor_question = pad_longest(batch_question)
            tensor_subtitles = pad_longest(batch_subtitles)
            
            list_tensor_answer = [ pad_longest(a) for a in batch_answers ]
            tensor_correct_index = torch.LongTensor([0])

            batch_correct_index = torch.LongTensor([0])
            if batch_correct_index[0] is not None:
                tensor_correct_index = torch.LongTensor(batch_correct_index)
            
            tensor_audio = torch.LongTensor([0])
            if use_audio:
                tensor_audio = torch.stack([pad_longest(list(v)) for v in zip(*batch_audio)]).permute(1, 0, 2)
            tensor_images = torch.LongTensor([0])
            if use_video:
                tensor_images = torch.stack([pad_longest(list(v)) for v in zip(*batch_images)]).permute(1, 0, 2)

            yield tensor_question, tensor_images, tensor_audio, tensor_subtitles, list_tensor_answer, tensor_correct_index, batch_indicator

if __name__ == "__main__":
    pass
