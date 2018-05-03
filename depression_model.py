import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import *

class Vocab:
    def __init__(self, embeddings_path, dimension = 300):
        self.vec = [[0]*dimension]
        self.word_count = 1
        self.ind2word = {0:"__unk__"}
        self.word2ind = {"__unk__":0}
        self.available_word2ind = {}

        self.tag_count = 0
        self.tag2id = {}
        self.id2tag = {}

        self.embeddings_path = embeddings_path
        self.dimension = dimension

        self.oov_counts = 0
        self.oovs = []

        self.hit_counts = 0
        self.hits = []

        self.load_vocab()

    def load_vocab(self):
        print "Loading vocab..."
        counter = 0
        for line in open(self.embeddings_path):
            values = line.strip().split(" ")
            if len(values) != (self.dimension + 1):
                continue
            v = []
            w = values[0].split("|")[0]
            self.available_word2ind[w] = counter
            counter += 1

    def load_vectors(self):
        print "Loading text vectors..."
        for line in open(self.embeddings_path):
            values = line.strip().split(" ")
            if len(values) != (self.dimension + 1):
                continue
            w = values[0]
            if w in self.word2ind:
                v = []
                for i in xrange(1, len(values)):
                    v.append(float(values[i]))
                self.vec.append(v)
        print "Done!"

    def update_hit(self, word):
        if word not in self.hits:
            self.hit_counts += 1
            self.hits.append(word)

    def update_oov(self, word):
        if word not in self.oovs:
            self.oov_counts += 1
            self.oovs.append(word)

    def get_word_id(self, word):
        word = word.lower()
        if word in self.word2ind:
            self.update_hit(word)
            return self.word2ind[word]
        elif word in self.available_word2ind:
            self.ind2word[self.word_count] = word
            self.word2ind[word] = self.word_count
            self.word_count += 1
            self.update_hit(word)
            return self.word2ind[word]
        else:
            self.update_oov(word)
            return 0

    def get_tag_id(self, tag):
        if tag not in self.tag2id:
            self.tag2id[tag] = self.tag_count
            self.id2tag[self.tag_count] = tag
            self.tag_count += 1
        return self.tag2id[tag]


class Attn(nn.Module):
    def __init__(self, hidden_size, use_gpu=False, max_length=MAX_LENGTH):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        self.use_gpu = use_gpu
        self.lin = nn.Linear(self.hidden_size * 2, hidden_size * 2)
        p_tensor = torch.zeros(1, hidden_size * 2)
        if use_gpu:
            p_tensor = p_tensor.cuda()
        self.weight_vec = nn.Parameter(p_tensor)

    def forward(self, outputs):
        seq_len = len(outputs)

        if self.use_gpu:
            attn_energies = Variable(torch.zeros(seq_len).cuda())  # B x 1 x S
        else:
            attn_energies = Variable(torch.zeros(seq_len))  # B x 1 x S

        for i in range(seq_len):
            attn_energies[i] = self.score(outputs[i])
        return F.softmax(attn_energies).unsqueeze(0).unsqueeze(0)

    def score(self, output):
        energy = self.lin(output)
        energy = torch.dot(self.weight_vec.view(-1), energy.view(-1))
        return energy


class DepressionModel(nn.Module):
    def __init__(self, vocab, args):
        super(DepressionModel, self).__init__()
        # self.input_size = input_size
        self.hidden_size = args.hidden_size
        self.n_layers = args.layers
        self.use_gpu = args.use_gpu

        embedding_size = len(vocab.vec[0])

        self.embedding = nn.Embedding(vocab.word_count, embedding_size)
        emb_tensor = to_device(torch.FloatTensor(vocab.vec), args.use_gpu)
        self.embedding.weight = nn.Parameter(emb_tensor)
        self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(embedding_size, self.hidden_size, self.n_layers, bidirectional=True)
        self.post_attn = Attn(self.hidden_size, use_gpu = self.use_gpu)
        self.timeline_attn = Attn(self.hidden_size, use_gpu = self.use_gpu)
        self.out = nn.Linear(self.hidden_size * 2, vocab.tag_count)

    def forward(self, input_batch):
        seq_lengths = to_device(torch.LongTensor(map(len, input_batch)), self.use_gpu)
        seq_tensor = to_device(Variable(torch.zeros((len(input_batch), seq_lengths.max()))), self.use_gpu).long()

        for idx, (seq, seqlen) in enumerate(zip(input_batch, seq_lengths)):
            seq_tensor[idx, :seqlen] = to_device(torch.LongTensor(seq), self.use_gpu)

        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        seq_tensor = seq_tensor[perm_idx]
        seq_tensor = to_device(seq_tensor.transpose(0, 1), self.use_gpu)  # (B,L,D) -> (L,B,D)

        seq_tensor = self.embedding(seq_tensor)
        packed_input = pack_padded_sequence(seq_tensor, seq_lengths.cpu().numpy())

        packed_output, (ht, ct) = self.lstm(packed_input)

        post_rnn_outputs, _ = pad_packed_sequence(packed_output)
        # output is now L,B,D

        post_rnn_outputs = post_rnn_outputs.transpose(0, 1) # (L,B,D) -> (B,L,D)

        post_attn_outputs = [None] * len(post_rnn_outputs)
        for i in xrange(0, len(post_rnn_outputs)):
            post_attn_weights = self.post_attn(post_rnn_outputs[i])
            post_rnn_outputs_T = post_rnn_outputs[i].transpose(0, 1)
            post_attn_weights_T = post_attn_weights.view(1, -1).transpose(0, 1)
            post_attn_outputs[perm_idx[i]] = torch.matmul(post_rnn_outputs_T, post_attn_weights_T).transpose(0, 1).squeeze()

        timeline_attn_weights = self.timeline_attn(post_attn_outputs)
        post_attn_outputs_stacked_T = torch.stack(post_attn_outputs).transpose(0, 1)
        timeline_attn_weights_T = timeline_attn_weights.view(1, -1).transpose(0, 1)
        print post_attn_outputs_stacked_T.data.shape
        print timeline_attn_weights_T.data.shape
        final_attn_output = torch.matmul(post_attn_outputs_stacked_T, timeline_attn_weights_T)

        final_output = self.out(final_attn_output.transpose(0, 1))

        return final_output