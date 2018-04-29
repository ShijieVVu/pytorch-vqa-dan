import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchtext.vocab as vocab

from model import Classifier

class TextEncoder(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_size):
        super(TextEncoder, self).__init__()
        self.embed = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.bilstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, 
                                batch_first=False, 
                                bidirectional=True,
                                dropout=0.5)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        embed_output = self.embed(x)
        bilstm_output, _ = self.bilstm(self.dropout(embed_output))
        return bilstm_output

    def load_pretrained(self, dictionary):
        print("Loading pretrained weights...")
        # Load pretrained vectors for embedding layer
        glove = vocab.GloVe(name='6B', dim=self.embed.embedding_dim)

        # Build weight matrix here
        pretrained_weight = self.embed.weight.data
        for word, idx in dictionary.items():
            if word.lower() in glove.stoi:     
                vector = glove.vectors[ glove.stoi[word.lower()] ]
                pretrained_weight[ idx ] = vector

        self.embed.weight = nn.Parameter(pretrained_weight)       

class AnswerEncoder(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_size):
        super(AnswerEncoder, self).__init__()
        self.embed = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, 
                                batch_first=False, dropout=0.5)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        embed_output = self.embed(x)
        output, _ = self.lstm(self.dropout(embed_output))
        last_output = output[-1]
        return last_output

    def load_pretrained(self, dictionary):
        print("Loading pretrained weights...")
        # Load pretrained vectors for embedding layer
        glove = vocab.GloVe(name='6B', dim=self.embed.embedding_dim)

        # Build weight matrix here
        pretrained_weight = self.embed.weight.data
        for word, idx in dictionary.items():
            if word.lower() in glove.stoi:     
                vector = glove.vectors[ glove.stoi[word.lower()] ]
                pretrained_weight[ idx ] = vector

        self.embed.weight = nn.Parameter(pretrained_weight)  

class Attention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Attention, self).__init__()

        memory_size = 2 * hidden_size
        self.W = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.Wm = nn.Linear(in_features=memory_size, out_features=hidden_size)
        self.Wh = nn.Linear(in_features=hidden_size, out_features=1)

        self.dropout = nn.Dropout(p=0.5)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

    def forward(self, feature, memory):
         # (seq_len, batch_size, dim) * (batch_size, dim)
        h = self.tanh(self.W(self.dropout(feature))) * self.tanh(self.Wm(self.dropout(memory)))
        # attention weights for text features
        alpha = self.softmax(self.Wh(self.dropout(h)))  # (seq_len, batch_size, memory_size)
        return alpha

class rDAN(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_size, answer_size, k=2):
        super(rDAN, self).__init__()

        # Build Text Encoder
        self.textencoder = TextEncoder(num_embeddings=num_embeddings, 
                                       embedding_dim=embedding_dim, 
                                        hidden_size=hidden_size)

        memory_size = 2 * hidden_size # bidirectional
        
        # Visual Attention
        self.attnV = Attention(2048, hidden_size)
        self.P = nn.Linear(in_features=2048, out_features=memory_size)
    
        # Textual Attention
        self.attnU = Attention(2*hidden_size, hidden_size)

        # Scoring Network
        self.classifier = Classifier(memory_size, hidden_size, answer_size, 0.5)

        # Dropout
        self.dropout = nn.Dropout(p=0.5)

        # Activations
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(0) # Softmax over first dimension

        # Loops
        self.k = k

    def forward(self, visual, text):

        batch_size = visual.shape[0]

        # Prepare Visual Features
        visual = visual.view(batch_size, 2048, -1)
        vns = visual.permute(2,0,1) # (nregion, batch_size, dim)

        # Prepare Textual Features
        text = text.permute(1,0)
        uts = self.textencoder.forward(text) # (seq_len, batch_size, dim)

        # Initialize Memory
        u = uts.mean(0)
        v = self.tanh( self.P( vns.mean(0) ))
        memory = v * u

        # K indicates the number of hops
        for k in range(self.k):
            # Compute Visual Attention
            alphaV = self.attnV(vns, memory)
            # Sum over regions
            v = self.tanh(self.P(alphaV * vns)).sum(0)

            # Text 
            # (seq_len, batch_size, dim) * (batch_size, dim)
            alphaU = self.attnU(uts, memory)
            # Sum over sequence
            u = (alphaU * uts).sum(0) # Sum over sequence
            
            # Build Memory
            memory = memory + u * v
        
        # We compute scores using a classifier
        scores = self.classifier(memory)

        return scores

class SubtitleEncoder(nn.Module):
    def __init__(self, embedding):
        super(SubtitleEncoder, self).__init__()
        # Define CNN Network Parmas here
        self.embed = embedding
        self.conv1 = nn.Conv2d(1, 4, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(4, 9, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, subtitles):
        # Define Forward params here
        x0 = subtitles.permute(1, 0)
        x1 = self.embed(x0)
        x2 = self.conv1(x1.unsqueeze(1))
        x3 = F.relu(self.pool1(x2))
        x4 = self.conv2(x3)
        x5 = self.dropout(F.relu(self.pool2(x4)))
        x6 = x5.permute(2, 0, 1, 3)
        bs = subtitles.size()[1]
        x7 = x6.contiguous().view(x6.size()[0], bs, -1)
        return x7

class AudioEncoder(nn.Module):
    def __init__(self):
        super(AudioEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(4, 9, 5)
        self.pool2 = nn.MaxPool2d(2, 2)

    def forward(self, audios):
        x0 = audios.permute(2, 0, 1).float()
        x1 = self.conv1(x0.unsqueeze(1))
        x2 = self.pool1(x1)
        x3 = self.conv2(x2)
        x4 = self.pool2(x3)
        x5 = x4.permute(2, 0, 1, 3)
        bs = audios.size()[2]
        x6 = x5.contiguous().view(x5.size()[0], bs, -1)
        return x6

class VideoEncoder(nn.Module):
    def __init__(self):
        super(VideoEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(4, 9, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(9, 16, 5)
        self.pool3 = nn.MaxPool2d(2, 2)

    def forward(self, images):
        x = images.permute(2, 0, 1).float()
        x = self.conv1(x.unsqueeze(1))
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        '''
        x = self.conv3(x)
        x = self.pool3(x)
        '''
        x = x.permute(2, 0, 1, 3)
        bs = images.size()[2]
        x = x.contiguous().view(x.size()[0], bs, -1)
        return x
        
        
class ScoreModel(nn.Module):
    def __init__(self, hidden_size):
        super(ScoreModel, self).__init__()
        '''
        self.fc1 = nn.Linear(in_features=3 * hidden_size, out_features=hidden_size)
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.fc3 = nn.Linear(in_features=hidden_size, out_features=1)
        '''
        self.fc = nn.Sequential(
            nn.Linear(3*hidden_size, 1),
            )
    def forward(self, memory, answer_feature):
        scores_options = []
        for answer in answer_feature:
            score = F.relu(self.fc(torch.cat([memory, answer])))
            scores_options.append(score)
        scores = torch.stack(scores_options)
        return scores

class MovieDAN(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_size, answer_size, k=2):
        super(MovieDAN, self).__init__()
         # Build Text Encoder
         # This encoder will encode all text
        self.textencoder = TextEncoder(num_embeddings=num_embeddings, 
                                       embedding_dim=embedding_dim, 
                                        hidden_size=hidden_size)

	# Share Embedding Matrix
        self.subtitleencoder = SubtitleEncoder(self.textencoder.embed)
        self.audioencoder = AudioEncoder()
        self.videoencoder = VideoEncoder()

        memory_size = 2 * hidden_size # bidirectional
        
        # Visual Attention
#        self.attnV = Attention(2048, hidden_size)
        self.Ps = nn.Linear(in_features=261, out_features=memory_size)
        self.Pa = nn.Linear(in_features=261, out_features=memory_size)
        self.Pv = nn.Linear(in_features=1125, out_features=memory_size)
    
        # Question Attention
        self.attnQ = Attention(memory_size, hidden_size)

        # Subtitle Attention
        self.attnS = Attention(261, hidden_size)

        # Audio Attention
        self.attnA = Attention(261, hidden_size)

        # Video Attention
        self.attnV = Attention(1125, hidden_size)

        # Answer Encoder
        self.answerencoder = AnswerEncoder(num_embeddings=num_embeddings, 
                                            embedding_dim=embedding_dim, 
                                            hidden_size=hidden_size)

        self.answerencoder.embed = self.textencoder.embed 
        # Memory & Answer Scoring
        self.scoring = nn.Bilinear(memory_size, hidden_size, 1)
        
        # Dropout
        self.dropout = nn.Dropout(p=0.5)

        # Activations
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(0) # Softmax over first dimension

        # Loops
        self.k = k

        # scoring model
        self.score_model = ScoreModel(hidden_size)

    def forward(self, question, images, audios, subtitles, list_answers):
        # Prepare Question Features
        qts = self.textencoder.forward(question) # (seq_len, batch_size, dim)
        sts = self.subtitleencoder.forward(subtitles) #
        ats = self.audioencoder.forward(audios)
        vts = self.videoencoder.forward(images)

        a_qs = 1/3
        a_qa = 1/3
        a_qv = 1/3
        
        # Initialize Memory
        q = qts.mean(0)
        s = self.tanh(self.Ps(sts.mean(0)))
        a = self.tanh(self.Pa(ats.mean(0)))
        v = self.tanh(self.Pv(vts.mean(0)))
        memory = a_qs * q * s + a_qa * q * a + a_qv * q * v

        # K indicates the number of hops
        for k in range(self.k):
            
            # Question Attention
            alphaQ = self.attnQ(qts, memory)
            q = (alphaQ * qts).sum(0)

            # Subtitle Attention
            alphaS = self.attnS(sts, memory)
            s = (self.tanh(self.Ps(alphaS * sts))).sum(0)

            # Audio Attention
            alphaA = self.attnA(ats, memory)
            a = (self.tanh(self.Pa(alphaA * ats))).sum(0)

            # Video Attention
            alphaV = self.attnV(vts, memory)
            v = (self.tanh(self.Pv(alphaV * vts))).sum(0)
   
            # Build Memory
            memory = a_qs * q * s + a_qa * q * a + a_qv * q * v

        # ( batch_size, memory_size )
        # We compute scores using a classifier
        list_answer_features = []
        for answers in list_answers:
            features = self.answerencoder.forward(answers)
            list_answer_features.append(features)
              
        answer_features = torch.stack(list_answer_features) #(batch_size, answer_size, hidden_size)

        scores_list = []
        for idx, answer_feature in enumerate(answer_features, 0):
            score = self.score_model(memory[idx], answer_feature)
            scores_list.append(score)
        scores = torch.squeeze(torch.stack(scores_list))
        return scores

if __name__ == "__main__":
    "Hello Shijie"
    
