# Name for model and directory
name = 'audio_model'

# paths
video_path = '/media/shijie/Users/WUSHI/github/Multiple-Attention-Model-for-MovieQA/data/data_processed/' # path to the processed video file folder
video_postfix = '.video.mp4features.p' # name postfix of each video feature file
audio_path = '/home/shijie/Downloads/features/sound_out_all/conv_16/tf_feat_' # path to the processed audio file folder
audio_postfix = '.video_16.npy' # name postfix of each audio feature file

# preprocess config
preprocess_batch_size = 64
image_size = 448  # scale shorter end of image to this size and centre crop
output_size = image_size // 32  # size of the feature maps after processing through a network
output_features = 2048  # number of feature maps thereof
central_fraction = 0.875  # only take this much of the centre when scaling and centre cropping

# model config
pretrained = True
embedding_dim = 128
hidden_size = 64
max_answers = 3000
movie_answer_size = 5

# training config
epochs = 50
batch_size = 8
initial_lr = 1e-3  # default Adam lr
lr_halflife = 50000  # in iterations
data_workers = 16

# prediction config
use_prior_model = True
prior_weight_path = "/home/shijie/github/pytorch-vqa-dan/model_sub_model/dan-audio-E04-A0.302.pt"
pred_file_path = "./prediction/test-models"

# model specification
sub_out = 261
audio_out = 549
video_out = 1125

# Weights on features
weight_qv = 0
weight_qs = 1
weight_qa = 0

k = 2
