import string
import re
import warnings
from argparse import ArgumentParser
from collections import Counter

warnings.filterwarnings("ignore")

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from grokfast import *


def L2(model):
    L2_ = 0.
    for p in model.parameters():
        L2_ += torch.sum(p**2)
    return L2_


def rescale(model, alpha):
    for p in model.parameters():
        p.data = alpha * p.data


class SentimentRNN(nn.Module):
    def __init__(self, no_layers, vocab_size, hidden_dim, embedding_dim, output_dim, drop_prob=0.0):
        super(SentimentRNN,self).__init__()

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.no_layers = no_layers
        self.vocab_size = vocab_size

        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        #lstm
        self.lstm = nn.LSTM(input_size=embedding_dim,hidden_size=self.hidden_dim,
                            num_layers=no_layers, batch_first=True)

        # dropout layer
        self.dropout = nn.Dropout(drop_prob)

        # linear and sigmoid layer
        self.fc = nn.Linear(self.hidden_dim, output_dim)
        self.sig = nn.Sigmoid()

        self.register_buffer('device_checker', torch.zeros(0), False)

    def forward(self,x,hidden):
        batch_size = x.size(0)
        # embeddings and lstm_out
        embeds = self.embedding(x)  # shape: B x S x Feature   since batch = True
        #print(embeds.shape)  #[50, 500, 1000]
        lstm_out, hidden = self.lstm(embeds, hidden)

        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim) 

        # dropout and fully connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)

        # sigmoid function
        sig_out = self.sig(out)

        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)

        sig_out = sig_out[:, -1] # get last batch of labels

        # return last sigmoid output and hidden state
        return sig_out, hidden

    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        h0 = torch.zeros((self.no_layers, batch_size, self.hidden_dim)).to(self.device_checker.device)
        c0 = torch.zeros((self.no_layers, batch_size, self.hidden_dim)).to(self.device_checker.device)
        hidden = (h0, c0)
        return hidden


# function to predict accuracy
def acc(pred, label):
    pred = torch.round(pred.squeeze())
    return torch.sum(pred == label.squeeze()).item()


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    alpha = args.init_scale
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_csv = './IMDB-Dataset.csv'
    df = pd.read_csv(base_csv)

    X, y = df[:args.size]['review'].values, df[:args.size]['sentiment'].values
    x_train, x_test, y_train, y_test = train_test_split(X, y, stratify=y)

    def preprocess_string(s):
        # Remove all non-word characters (everything except numbers and letters)
        s = re.sub(r"[^\w\s]", '', s)
        # Replace all runs of whitespaces with no space
        s = re.sub(r"\s+", '', s)
        # replace digits with no space
        s = re.sub(r"\d", '', s)

        return s

    def tockenize(x_train, y_train, x_val, y_val):
        word_list = []

        stop_words = {
            'a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all', 'am', 'an',
            'and', 'any', 'are', 'aren', "aren't", 'as', 'at', 'be', 'because', 'been',
            'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'couldn', "couldn't",
            'd', 'did', 'didn', "didn't", 'do', 'does', 'doesn', "doesn't", 'doing', 'don',
            "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadn',
            "hadn't", 'has', 'hasn', "hasn't", 'have', 'haven', "haven't", 'having', 'he', 'her',
            'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
            'into', 'is', 'isn', "isn't", 'it', "it's", 'its', 'itself', 'just', 'll',
            'm', 'ma', 'me', 'mightn', "mightn't", 'more', 'most', 'mustn', "mustn't", 'my',
            'myself', 'needn', "needn't", 'no', 'nor', 'not', 'now', 'o', 'of', 'off',
            'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over',
            'own', 're', 's', 'same', 'shan', "shan't", 'she', "she's", 'should', "should've",
            'shouldn', "shouldn't", 'so', 'some', 'such', 't', 'than', 'that', "that'll", 'the',
            'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', 'this', 'those',
            'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was', 'wasn',
            "wasn't", 'we', 'were', 'weren', "weren't", 'what', 'when', 'where', 'which', 'while',
            'who', 'whom', 'why', 'will', 'with', 'won', "won't", 'wouldn', "wouldn't", 'y',
            'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves',
        }

        for sent in x_train:
            for word in sent.lower().split():
                word = preprocess_string(word)
                if word not in stop_words and word != '':
                    word_list.append(word)

        corpus = Counter(word_list)
        # sorting on the basis of most common words
        corpus_ = sorted(corpus,key=corpus.get,reverse=True)[:1000]
        # creating a dict
        onehot_dict = {w:i+1 for i,w in enumerate(corpus_)}

        def _padding(sentences, seq_len):
            features = np.zeros((len(sentences), seq_len), dtype=int)
            for i, review in enumerate(sentences):
                if len(review) != 0:
                    features[i, -len(review):] = np.array(review)[:seq_len]
            return features

        # tockenize
        final_list_train, final_list_test = [], []
        for sent in x_train:
            final_list_train.append([onehot_dict[preprocess_string(word)] for word in sent.lower().split() 
                                     if preprocess_string(word) in onehot_dict.keys()])
        final_list_train = _padding(final_list_train, 500)
        for sent in x_val:
            final_list_test.append([onehot_dict[preprocess_string(word)] for word in sent.lower().split() 
                                    if preprocess_string(word) in onehot_dict.keys()])
        final_list_test = _padding(final_list_test, 500)

        encoded_train = [1 if label =='positive' else 0 for label in y_train]  
        encoded_test = [1 if label =='positive' else 0 for label in y_val] 
        return np.array(final_list_train), np.array(encoded_train), np.array(final_list_test), np.array(encoded_test), onehot_dict

    # create Tensor datasets
    x_train, y_train, x_test, y_test, vocab = tockenize(x_train, y_train, x_test, y_test)
    train_data = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    valid_data = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))

    # dataloaders
    train_loader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size)
    valid_loader = DataLoader(valid_data, shuffle=True, batch_size=args.batch_size)

    # obtain one batch of training data
    dataiter = iter(train_loader)
    sample_x, sample_y = next(dataiter)

    # define model
    no_layers = 2
    vocab_size = len(vocab) + 1 #extra 1 for padding
    embedding_dim = 64
    output_dim = 1
    hidden_dim = 256
    model = SentimentRNN(no_layers, vocab_size, hidden_dim, embedding_dim, output_dim, drop_prob=0.0)
    model.to(device)
    
    rescale(model, alpha)
    L2_ = L2(model)

    # loss and optimization functions
    criterion = nn.BCELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    #######

    train_loss_min = np.Inf
    valid_loss_min = np.Inf
    train_acc_max = 0.
    valid_acc_max = 0.

    # train for some number of epochs
    log_steps, train_accs, test_accs, test_losses, train_losses, train_avg_losses, test_avg_losses, train_avg_accs, test_avg_accs = [], [], [], [], [], [], [], [], []
    step, epoch = 0, 0

    grads = None

    pbar = tqdm(total=args.iterations)
    while step < args.iterations:
        epoch_loss = 0
        epoch_acc = 0
        test_epoch_loss = 0
        test_epoch_acc = 0
        train_size = 0
        test_size = 0

        for inputs, labels in train_loader:
            model.train()

            inputs, labels = inputs.to(device), labels.to(device)
            # initialize hidden state 
            h = model.init_hidden(inputs.shape[0])
            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            model.zero_grad()

            output, h = model(inputs, h)
            # calculate the loss and perform backprop
            loss = criterion(output.squeeze(), labels.float())
            train_size += inputs.shape[0]
            epoch_loss += loss.item() * inputs.shape[0]
            train_losses.append(loss.item())
            # calculating accuracy
            accuracy = acc(output, labels)
            train_acc = accuracy / inputs.shape[0]
            epoch_acc += accuracy
            train_accs.append(train_acc)

            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)

            #######

            trigger = False

            if args.filter == "none":
                pass
            elif args.filter == "ma":
                grads = gradfilter_ma(model, grads=grads, window_size=args.window_size, lamb=args.lamb, trigger=trigger)
            elif args.filter == "ema":
                grads = gradfilter_ema(model, grads=grads, alpha=args.alpha, lamb=args.lamb)
            else:
                raise ValueError(f"Invalid gradient filter type `{args.filter}`")

            #######

            optimizer.step()

            #######

            model.eval()

            inputs, labels = next(iter(valid_loader))
            inputs, labels = inputs.to(device), labels.to(device)

            val_h = model.init_hidden(args.batch_size)
            val_h = tuple([each.data for each in val_h])
            output, val_h = model(inputs, val_h)
            val_loss = criterion(output.squeeze(), labels.float())
            test_size += inputs.shape[0]
            test_epoch_loss += val_loss.item() * inputs.shape[0]
            test_losses.append(val_loss.item())

            accuracy = acc(output, labels)
            val_acc = accuracy / args.batch_size
            test_epoch_acc += accuracy
            test_accs.append(val_acc)

            if (step + 1) % 10 == 0:
                tqdm.write(f'step : {step} train_loss : {loss.item()} val_loss : {val_loss.item()}\n'
                           f'train_accuracy : {train_acc} val_accuracy : {val_acc}')

            step += 1
            pbar.update()

        log_steps.append(step)

        epoch_loss = epoch_loss / train_size
        train_avg_losses.append(epoch_loss)

        epoch_acc = epoch_acc / train_size
        train_avg_accs.append(epoch_acc)

        test_epoch_loss = test_epoch_loss / test_size
        test_avg_losses.append(test_epoch_loss)

        test_epoch_acc = test_epoch_acc / test_size
        test_avg_accs.append(test_epoch_acc)

        tqdm.write(f"Epochs: {epoch} | epoch avg. acc: {epoch_acc:.3f} | "
                   f"test avg. acc: {test_epoch_acc:.3f}")

        if (epoch + 1) % 100 == 0 or step == args.iterations - 1:

            title = (f"IMDb Binary Sentiment Analysis")

            plt.plot(np.arange(step), train_accs, label="train")
            plt.plot(np.arange(step), test_accs, label="val")
            plt.legend()
            plt.title(title)
            plt.xlabel("Optimization Steps")
            plt.ylabel("Accuracy")
            plt.xscale("log", base=10)
            plt.grid()
            plt.savefig(f"results/imdb_acc_{args.label}.png", dpi=150)
            plt.close()

            plt.plot(np.arange(step), train_losses, label="train")
            plt.plot(np.arange(step), test_losses, label="val")
            plt.legend()
            plt.title(title)
            plt.xlabel("Optimization Steps")
            plt.ylabel("BCE Loss")
            plt.xscale("log", base=10)
            plt.yscale("log", base=10)
            plt.grid()
            plt.savefig(f"results/imdb_loss_{args.label}.png", dpi=150)
            plt.close()

            torch.save({
                'its': np.arange(len(train_losses)),
                'its_avg': np.arange(len(train_avg_losses)),
                'train_acc': train_accs,
                'train_loss': train_losses,
                'train_avg_acc': train_avg_accs,
                'train_avg_loss': train_avg_losses,
                'val_acc': test_accs,
                'val_loss': test_losses,
                'val_avg_acc': test_avg_accs,
                'val_avg_loss': test_avg_losses,
            }, f"results/imdb_{args.label}.pt")

        epoch += 1


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--label", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--weight_decay", type=float, default=1.0)
    parser.add_argument("--gradient_clip", type=float, default=5.0)
    parser.add_argument("--size", type=int, default=1000)
    parser.add_argument("--init_scale", type=float, default=6.0) # init_scale 1.0 no grokking / init_scale 6.0 grokking

    # Grokfast
    parser.add_argument("--filter", type=str, choices=["none", "ma", "ema", "fir"], default="none")
    parser.add_argument("--alpha", type=float, default=0.99)
    parser.add_argument("--window_size", type=int, default=100)
    parser.add_argument("--lamb", type=float, default=5.0)
    args = parser.parse_args()

    model_suffix = f'size{args.size}_alpha{args.init_scale:.4f}'

    filter_str = ('_' if args.label != '' else '') + args.filter
    window_size_str = f'_w{args.window_size}'
    alpha_str = f'_a{args.alpha:.3f}'.replace('.', '')
    lamb_str = f'_l{int(args.lamb)}'

    if args.filter == 'none':
        filter_suffix = ''
    elif args.filter == 'ma':
        filter_suffix = window_size_str + lamb_str
    elif args.filter == 'ema':
        filter_suffix = alpha_str + lamb_str
    else:
        raise ValueError(f"Unrecognized filter type {args.filter}")

    optim_suffix = ''
    if args.weight_decay != 0:
        optim_suffix = optim_suffix + f'_wd{args.weight_decay:.1e}'.replace('.', '')
    if args.lr != 1e-3:
        optim_suffix = optim_suffix + f'_lrx{int(args.lr / 0.0003)}'

    args.label = args.label + model_suffix + filter_str + filter_suffix + optim_suffix
    print(f'Experiment results saved under name: {args.label}')

    main(args)
