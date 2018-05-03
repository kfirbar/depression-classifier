import json
import pandas as pd
import random
import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from collections import Counter
import pickle
import os

MAX_LENGTH = 300

def load_rsdd_texts(filepath, vocab, nlp, load_percentage, post_percentage):
    instances = []
    fp = open(filepath, "r")
    counter = 0
    for line in fp:
        if random.random() < load_percentage:
            try:
                data = json.loads(line)
            except ValueError:
                continue
            counter += 1
            label = data[0]["label"]
            instance = [vocab.get_tag_id(label)]
            posts = []
            for p in data[0]["posts"]:
                if random.random() < post_percentage:
                    proc_text = nlp(p[1].encode('ascii', errors='replace').decode('utf-8', errors="replace").strip())
                    if len(proc_text) > MAX_LENGTH:
                        continue
                    input = [vocab.get_word_id(token.text) for token in proc_text]
                    posts.append(input)
            instance.append(posts)
            instances.append(instance)
            #return instances
            if counter % 10 == 0:
                print "Reading %d line" % counter
    return instances

def to_device(t, use_gpu):
    if use_gpu:
        t = t.cuda()
    return t

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def serialize(model, lang, filename):
    torch.save(model.state_dict(), filename + ".model")
    pickle.dump(lang, open(filename + ".lang.p", "wb"))


def save_train_test(train, test, lang, folder):
    counter = 0
    for d in train:
        input = d[2]
        target = lang.id2tag[d[1].data.cpu().numpy()[0]]
        if not os.path.exists(folder + "/train/" + target):
            os.makedirs(folder + "/train/" + target)
        f = open(folder + "/train/" + target + "/" + str(counter) + ".txt", "w")
        f.write(input)
        f.close()
        counter += 1

    f_test = open(folder + "/test.txt", "w")
    f_labels = open(folder + "/test_labels.txt", "w")
    for d in test:
        input = d[2]
        input = input.replace('\r', '').replace('\n', '')
        target = lang.id2tag[d[1].data.cpu().numpy()[0]]
        f_labels.write(target + "\n")
        f_test.write(input + "\n")
        counter += 1
    f_test.close()
    f_labels.close()
