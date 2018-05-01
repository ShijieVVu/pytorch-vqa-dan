import sys
import os.path
import math
import json
import gc
import resource

import torch
# Monkey-patch because I trained with a newer version.
# This can be removed once PyTorch 0.4.x is out.
# See https://discuss.pytorch.org/t/question-about-rebuild-tensor-v2/14560
import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2
    
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from tqdm import tqdm

import config
import movie
from dan import TextEncoder, MovieDAN
import utils

torch.backends.cudnn.enabled = False

# path to the weight file
weight_path = '/home/shijie/github/pytorch-vqa-dan/dan-audio-E11-A0.321.pt'

# path of the output file name
file_name = 'val_a_conv14_acc.txt'

def update_learning_rate(optimizer, iteration):
    lr = config.initial_lr * 0.5**(float(iteration) / config.lr_halflife)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def run(net, dataset, optimizer, train=False, prefix='', epoch=0):
    """ Run an epoch over the given loader """
    if train:
        net.train()
    else:
        net.eval()

    niter = int(len(dataset) / dataset.batch_size)
    tq = tqdm(dataset.loader(), desc='{} E{:03d}'.format(prefix, epoch), ncols=0, total=niter)

    criterion = nn.CrossEntropyLoss().cuda()
   
    total_count = 0
    total_acc = 0
    total_loss = 0
    total_iterations = 0
    
    for q, v, au, s, la, qids, c in tq:
#        import pdb; pdb.set_trace()
        var_params = {
            'requires_grad': False,
        }
        q = Variable(q.cuda())
        v = Variable(v.cuda())
        au = Variable(au.cuda())
       	s = Variable(s.cuda())
        la = [ Variable(a.cuda()) for a in la ]
        c = Variable(c.cuda()) # correct answers

        out = net(q, v, au, s, la)
        
        loss = criterion(out, c)

        # Compute our own accuracy
#        import pdb; pdb.set_trace()

        _, pred = out.data.max(dim=1)
        acc = (pred == c.data).float()

        with open(file_name, 'a+') as f:
            for (id, p) in zip(qids, pred):
                f.write("{} {}\n".format(id, p))

        total_iterations += 1


        total_count += acc.shape[0]
        total_loss += loss.data[0] * acc.shape[0]
        total_acc += acc.sum()

    acc = total_acc / total_count
    loss = total_loss / total_count
    print("loss: {} acc {}".format(loss, acc))
    return acc


def main():
    if len(sys.argv) > 1:
        name = ' '.join(sys.argv[1:])
    else:
        from datetime import datetime
        name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    target_name = os.path.join('logs', '{}.pth'.format(name))
    print('will save to {}'.format(target_name))

    cudnn.benchmark = True

    train_dataset = movie.get_dataset(train=True, use_subtitle=config.use_subtitle, use_audio=config.use_audio, use_video=config.use_video)
    val_dataset = movie.get_dataset(val=True, use_subtitle=config.use_subtitle, use_audio=config.use_audio, use_video=config.use_video)
    test_dataset = movie.get_dataset(test=True, use_subtitle=config.use_subtitle, use_audio=config.use_audio, use_video=config.use_video)
    
    # Build Model
    vocab_size = len(train_dataset.vocab)
    model = MovieDAN(num_embeddings=vocab_size,
                     embedding_dim=config.embedding_dim,
                     hidden_size=config.hidden_size, 
                     answer_size=config.movie_answer_size,
                     weight_qv=config.weight_qv,
                     weight_qs=config.weight_qs,
                     weight_qa=config.weight_qa,
                     sub_out=config.sub_out,
                     audio_out=config.audio_out,
                     video_out=config.video_out,
                     k=config.k
    )

#    import pdb; pdb.set_trace()
    #weights = torch.load('./model_1s/dan-audio-E01-A0.252.pt')
    weights = torch.load(weight_path)
    #model = weights
    model.load_state_dict(weights)
    net = nn.DataParallel(model).cuda()
    optimizer = optim.Adam([p for p in net.parameters() if p.requires_grad])
    config_as_dict = {k: v for k, v in vars(config).items() if not k.startswith('__')}
    run(net, val_dataset, optimizer, train=False, prefix='test', epoch=0)

if __name__ == '__main__':
    main()
