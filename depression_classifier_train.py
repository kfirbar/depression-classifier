import spacy as sp
import pandas as pd
import numpy as np
import spacy as sp
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import random
import torch.nn.functional as F
from random import shuffle
from sklearn.metrics import precision_recall_fscore_support
import pickle
from struct import Struct
import mmap
import argparse
import os
from collections import Counter
from depression_model import Vocab
from depression_model import DepressionModel
from utils import *

def train_instance(instance, criterion, model, optimizer, use_gpu):
    optimizer.zero_grad()
    loss = 0
    output = model(instance[1])
    target_var = Variable(to_device(torch.LongTensor([long(instance[0])]), use_gpu))

    loss = criterion(output.view(1, -1), target_var)
    loss.backward()
    optimizer.step()
    return output, loss.data[0]

def eval(model, test):
    y_true = []
    y_predicted = []
    for t in test:
        input_var = t[0]
        target_var = t[1]
        output = model(input_var)
        y_predicted.append(np.argmax(softmax(output.data.cpu().numpy())))
        y_true.append(target_var.data.cpu().numpy()[0])
    return precision_recall_fscore_support(y_true, y_predicted, average='weighted')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="the train file")
    parser.add_argument("--user_percentage", help="the percentage amount of users to load from RSDD", type=float, default=0.1)
    parser.add_argument("--post_percentage", help="the percentage amount of posts to load for each user", type=float, default=0.1)
    parser.add_argument("--model", help="the model file to start from (a.k.a last checkpoint)")
    parser.add_argument("--vocab", help="the vocab (a.k.a last checkpoint lang file) to start from")
    parser.add_argument("--output", help="the output file for keeping the model")
    parser.add_argument("--seed", help="seed for randomization", type=int, default=100)
    parser.add_argument("--embeddings_path", help="the embeddings path to use")
    parser.add_argument("--embeddings_dimension", help="the embeddings dimensions", type=int, default=300)
    parser.add_argument("--hidden_size", help="the hidden size of the network", type=int, default=800)
    parser.add_argument("--layers", help="the number of LSTM layers", type=int, default=2)
    parser.add_argument("--epochs", help="the number of epochs to run", type=int, default=10000)
    parser.add_argument("--first_epoch_number", help="the number of the first epoch", type=int, default=1)
    parser.add_argument("--print_every", help="print every this number of batches", type=int, default=10)
    parser.add_argument("--sample_every", help="sample every this number of batches", type=int, default=20)
    parser.add_argument("--eval_every", help="sample every this number of batches", type=int, default=2500)
    parser.add_argument("--checkpoint_every", help="checkpoint every this number of batches", type=int, default=2)
    parser.add_argument("--use_gpu", help="run on GPU", type=bool, default=False)

    args = parser.parse_args()
    print "Starting..."
    vocab = Vocab(args.embeddings_path, args.embeddings_dimension)
    print "Loading Spacy..."
    nlp = sp.load('en')

    random.seed(args.seed)
    instances = load_rsdd_texts(args.train, vocab, nlp, args.user_percentage, args.post_percentage)
    vocab.load_vectors()
    print "Done!"
    print "Number of instances:", len(instances)
    print "hits:", vocab.hit_counts, "oovs:", vocab.oov_counts

    # shuffle(instances)
    # thresh = int(len(instances) * 0.9)
    # train = instances[0:thresh]
    # test = instances[thresh:]

    #print "Train size: ", len(train), "Test size:", len(test)

    model = DepressionModel(vocab, args)
    if args.use_gpu:
        model = model.cuda()
    print "Model is ready"

    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.0001
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    print "Optimizer is ready"
    loss = 0
    instance_counter = 0
    for e in range(1, args.epochs + 1):
        for i in xrange(0, len(instances)):
            instance_counter += 1
            output, iter_loss = train_instance(instances[i], criterion, model, optimizer, args.use_gpu)
            loss += iter_loss
            if instance_counter % args.print_every == 0:
                loss = loss / args.print_every
                print 'Epoch %d, instance %d, Current Loss = %.4f' % (e, i, loss)
                loss = 0
            if instance_counter % args.sample_every == 0:
                print "Target:", instances[i][0]
                print "Output:", np.argmax(softmax(output.data.cpu().numpy()))
            # if e % args.eval_every == 0:
            #     print "Eval:", eval(model, test)

    #print "Final eval:", eval(model, test)
    print "Done! saving model to ", args.output
    serialize(model, vocab, args.output)

if __name__ == "__main__":
    main()

