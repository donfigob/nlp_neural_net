# models.py

import csv
import time
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from typing import List
from sentiment_data import *
from utils import *


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str]) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :return:
        """
        return [self.predict(ex_words) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str]) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.)
    """
    def __init__(self, network, word_embeddings):
        self.dan = network
        self.embeddings = word_embeddings
        # raise NotImplementedError
    
    def predict(self, ex_words):
        indexer  = self.embeddings.word_indexer
        unk_index = indexer.index_of("UNK")
        indexes = []
        for word in ex_words:
            index = indexer.index_of(word)
            if index == -1:
                index = unk_index
            indexes.append(index)
        input_indixes = torch.tensor(indexes, dtype=torch.long)
        
        self.dan.eval()
        with torch.no_grad():
            log_probs = self.dan(input_indixes)
            prediction = torch.argmax(log_probs).item()
        return prediction

            




class DeepAveragingNetwork(nn.Module):
    def __init__(self, word_embeddings, hidden_size):
        super(DeepAveragingNetwork,self).__init__()

        self.embedding_layer = word_embeddings.get_initialized_embedding_layer()
        self.embedding_dim = word_embeddings.get_embedding_length()

        self.hidden_layer = nn.Linear(self.embedding_dim, hidden_size)
        self.activation = nn.ReLU()
        self.output_layer = nn.Linear(hidden_size, 2)
        self.log_softmax = nn.LogSoftmax(dim=0)

        nn.init.xavier_uniform_(self.hidden_layer.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, input_indixes):
        sentence_embeddings = self.embedding_layer(input_indixes)
        sentence_average = sentence_embeddings.mean(dim=0)
        return self.log_softmax(self.output_layer(self.activation(self.hidden_layer(sentence_average))))


def evaluate_dan(dan, examples, word_embeddings):
    indexer = word_embeddings.word_indexer
    unk_index = indexer.index_of("UNK")

    total_loss = 0.0
    num_correct = 0

    dan.eval()
    with torch.no_grad():
        for ex in examples:
            indexes = []
            for word in ex.words:
                index = indexer.index_of(word)
                if index == -1:
                    index = unk_index
                indexes.append(index)
            input_indixes = torch.tensor(indexes, dtype=torch.long)

            log_probs = dan(input_indixes)
            total_loss += (-log_probs[ex.label]).item()

            prediction = torch.argmax(log_probs).item()
            if prediction == ex.label:
                num_correct += 1

    return total_loss / len(examples), num_correct / len(examples)





def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample], word_embeddings: WordEmbeddings) -> NeuralSentimentClassifier:
    """
    Main entry point for your deep averaging network model.
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :return: A trained NeuralSentimentClassifier model
    """
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    indexer  = word_embeddings.word_indexer
    unk_index = indexer.index_of("UNK")
    pad_index = indexer.index_of("PAD")

    dan = DeepAveragingNetwork(word_embeddings,args.hidden_size)
    optimizer = optim.Adam(dan.parameters(), lr=args.lr)
    history = []

    for epoch in range(0, args.num_epochs):
        dan.train()
        train_start = time.perf_counter()
        total_loss =  0.0
        random.shuffle(train_exs)
        for start in range(0, len(train_exs), args.batch_size):
            batch = train_exs[start:start + args.batch_size]
            sentences = []
            labels = []

            for ex in batch:
                indexes = []
                for word in ex.words:
                    index = indexer.index_of(word)
                    if index == -1:
                        index = unk_index
                    indexes.append(index)

                sentences.append(torch.tensor(indexes, dtype=torch.long))
                labels.append(ex.label)

            input_indices = nn.utils.rnn.pad_sequence(
                sentences,
                batch_first=True,
                padding_value=pad_index,
            )
            labels = torch.tensor(labels, dtype=torch.long)

            # Next: adapt forward and the loss to process a complete batch.
            raise NotImplementedError(
                "Falta adaptar forward y la pérdida para entrenar con batches."
            )
        train_seconds = time.perf_counter() - train_start

        eval_start = time.perf_counter()
        train_loss, train_acc = evaluate_dan(dan, train_exs, word_embeddings)
        dev_loss, dev_acc = evaluate_dan(dan, dev_exs, word_embeddings)
        eval_seconds = time.perf_counter() - eval_start

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "dev_loss": dev_loss,
            "train_acc": train_acc,
            "dev_acc": dev_acc,
            "train_seconds": train_seconds,
            "eval_seconds": eval_seconds,
        })

        print(f"Epoch {epoch + 1}: " f"train loss={train_loss:.4f}, acc={train_acc:.2%}; " f"dev loss={dev_loss:.4f}, acc={dev_acc:.2%}")
        print("Total loss on epoch %i: %f" % (epoch, total_loss))

    columns = ["epoch", "train_loss", "dev_loss", "train_acc", "dev_acc",
               "train_seconds", "eval_seconds"]
    with open("metrics_baseline.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(history)

    return NeuralSentimentClassifier(dan, word_embeddings)

    # raise NotImplementedError
