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

class MovieDAN(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_size, answer_size, k=2):
        super(MovieDAN, self).__init__()
         # Build Text Encoder
         # This encoder will encode all text
        self.textencoder = TextEncoder(num_embeddings=num_embeddings, 
                                       embedding_dim=embedding_dim, 
                                        hidden_size=hidden_size)

        memory_size = 2 * hidden_size # bidirectional
        
        # Visual Attention
        self.attnV = Attention(2048, hidden_size)
        self.P = nn.Linear(in_features=2048, out_features=memory_size)
    
        # Question Attention
        self.attnQ = Attention(2*hidden_size, hidden_size)

        # Subtitle Attention
        self.attnS = Attention(2*hidden_size, hidden_size)

        # Answer Encoder
        self.answerencoder = AnswerEncoder(num_embeddings=num_embeddings, 
                                            embedding_dim=embedding_dim, 
                                            hidden_size=hidden_size)
        
        # Memory & Answer Scoring
        self.scoring = nn.Bilinear(memory_size, hidden_size, 1)
        
        # Dropout
        self.dropout = nn.Dropout(p=0.5)

        # Activations
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(0) # Softmax over first dimension

        # Loops
        self.k = k

    def forward(self, question, subtitles, list_answers):
        
        # Prepare Question Features
        qts = self.textencoder.forward(question) # (seq_len, batch_size, dim)

        sts = self.textencoder.forward(subtitles) #

        # Initialize Memory
        q = qts.mean(0)
        s = sts.mean(0)
        memory = q * s

        # K indicates the number of hops
        for k in range(self.k):
            
            # Question Attention
            alphaQ = self.attnQ(qts, memory)
            q = (alphaQ * qts).sum(0)

            # Subtitle Attention
            alphaS = self.attnS(sts, memory)
            s = (alphaS * sts).sum(0)
   
            # Build Memory
            memory = memory + q * s
        
        # ( batch_size, memory_size )
        # We compute scores using a classifier
        list_answer_features = []
        for answers in list_answers:
            features = self.answerencoder.forward(answers)
            list_answer_features.append(features)
        
      
        answer_features = torch.stack(list_answer_features) #(batch_size, answer_size, hidden_size)
      
        batch_size, memory_size = memory.shape
        batch_size, answer_size, hidden_size = answer_features.shape
        # memory: (batch_size, memory_size)
        # ( batch_size, hidden_size )
        # Bilinear scoring
        memory = memory.unsqueeze(1) # (batch_size, answer_size, memory_size)
        memory = memory.expand(batch_size, answer_size, memory_size)
        memory = memory.contiguous().view(-1, memory_size) # batch_size * answer_size, memory_size
        
        answer_features = answer_features.view(-1, hidden_size) # batch_size * answer_size, hidden_size)

        scores = self.scoring(memory.view(-1,memory_size), answer_features.view(-1, hidden_size))
        scores = scores.view(batch_size, answer_size)
        return scores

if __name__ == "__main__":
    pass
