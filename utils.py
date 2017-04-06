# from __future__ import unicode_literals
from collections import defaultdict
import numpy as np
import logging

import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame
import matplotlib as mpl
from matplotlib import rc
from matplotlib.backends.backend_pdf import PdfPages
import operator
import sys
# reload(sys)
# sys.setdefaultencoding("utf-8")

def init_logging(log_path, level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(module)s: %(message)s', datefmt='%m/%d/%Y %H:%M:%S')

    rootLogger = logging.getLogger()

    fileHandler = logging.FileHandler(log_path)
    fileHandler.setFormatter(formatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(formatter)
    rootLogger.addHandler(consoleHandler)

    rootLogger.setLevel(level)


def get_vocab(fname, top_k=20000, min_freq=1):
    wids = defaultdict(lambda: 0)
    freqs = defaultdict(lambda: 0)
    wids["<unk>"] = 0
    wids["<s>"] = 1
    wids["</s>"] = 2

    train_data = []
    with open(fname) as fin:
        for line in fin:
            words = line.strip().split(' ')
            for word in words:
                freqs[word] += 1
            train_data.append(["<s>"] + words + ["</s>"])

    # sorted_words will be a list of tuples
    sorted_words = sorted(freqs.iteritems(), key=operator.itemgetter(1), reverse=True)
    for i in range(top_k):
        word = sorted_words[i][0]
        if freqs[word] <= min_freq:
            #print("skipped word that occurs once")
            continue
        wids[word] = len(wids)
    id_to_words = {v:k for k, v in wids.iteritems()}

    id_train_data = [[wids[w] if w in wids else 0 for w in sent] for sent in train_data]
    return wids, id_to_words, id_train_data


def get_data(fname, vocab):
    data = []
    with open(fname) as fin:
        for line in fin:
            words = line.strip().split(' ')
            words = [vocab[w] for w in words]
            data.append([vocab["<s>"]] + words + [vocab["</s>"]])
    return data


def data_iterator(data_pair, batch_size):
    buckets = defaultdict(list)
    for pair in data_pair:
        src = pair[0]
        buckets[len(src)].append(pair)

    batches = []
    for src_len in buckets:
        bucket = buckets[src_len]
        np.random.shuffle(bucket)
        num_batches = int(np.ceil(len(bucket) * 1.0 / batch_size))
        for i in range(num_batches):
            cur_batch_size = batch_size if i < num_batches - 1 else len(bucket) - batch_size * i
            batches.append(([bucket[i * batch_size + j][0] for j in range(cur_batch_size)],
                           [bucket[i * batch_size + j][1] for j in range(cur_batch_size)]))

    np.random.shuffle(batches)
    for batch in batches:
        yield batch


def get_sent(sent, id_to_words):
    return [id_to_words[w] for w in sent]


def plot1(x, y, title=None, xlabel=None, ylabel=None, fname=None):
    pp = PdfPages("../plots/" + fname + ".pdf")
    fig = plt.figure()
    plt.plot(x, y, color='blue', marker='o', markersize=8)
    plt.title(title, fontsize=12)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    # plt.ylim(0)
    pp.savefig(fig)
    pp.close()


def heatmap(x, ind):
    pp = PdfPages("../plots/heatmap_en_de" + str(ind) + ".pdf")
    # mpl.rcParams['text.usetex'] = True
    # mpl.rcParams['text.latex.unicode'] = True
    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})

    rc('text', usetex=True)
    # sns.set(font="Arial Unicode MS")
    # sns.set_style({"savefig.dpi": 100})
    # plot it out
    plt.figure(figsize=(40, 70))
    sns.set(font_scale=3)
    ax = sns.heatmap(x, cmap=plt.cm.Purples, cbar_kws={"orientation": "vertical", "pad": 0.03})
    # set the x-axis labels on the top
    ax.xaxis.tick_top()
    # rotate the x-axis labels
    plt.xticks(rotation=40)
    plt.yticks(rotation=0)
    plt.margins(0.2)

    fig = ax.get_figure()
    fig.set_size_inches(20, 30)
    # get figure (usually obtained via "fig,ax=plt.subplots()" with matplotlib)
    # specify dimensions and save
    plt.tight_layout()
    pp.savefig(fig)
    pp.close()


def grab_data(file):
    d = []
    with open(file) as f:
        for line in f:
            d.append(float(line.strip()))
    return d
