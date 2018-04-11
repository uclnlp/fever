import numpy as np
from collections import defaultdict
from nltk.tokenize import wordpunct_tokenize as tokenize
from fever_io import load_split_trainset
import random

random.seed(42)

def calculate_accuracy(id2label:dict, id2prediction:dict, match_assert=True):
    """
    Takes two dictionaries: id->label; and id->prediction.
    Based on the id, label and prediction are matched, and accuracy is computed
    match_assert: boolean. Checks whether ids between the dicts overlap perfectly.
    """
    if match_assert:
        assert set(id2label.keys()) == set(id2prediction.keys())

    # how often is the prediction equal to the label, on average?
    accuracy = np.mean([id2prediction[id] == id2label[id] if id in id2prediction else False for id in id2label])
    print("Accuracy:", accuracy)
    return accuracy



def bigram_splitter(text:str):
    """
    Returns list of bigrams in given text.
    """
    tokens = tokenize(text)
    if len(tokens) < 2:
        bigrams = []
    else:
        bigrams = [tokens[i]+ " "+ tokens[i+1] for i in range(len(tokens)-1)]

    return bigrams


class NaiveBayesBow():
    def __init__(self, train_set, tokenisation_type="unigram", lowercase=True):
        """
        Defines and trains a Naive Bayes model from the given train_set, using
        bag-of-words features.
        NOTE: This assumes a balanced test set distribution.
        Inputs:
        - "tokenisation_type" can be changed to try unigram/bigram/both.
        - "lowercase" whether claims are lowercased before featurisation.
        """
        self.lowercase = lowercase

        # stores word counts for each label.
        label_to_word_freq_dict = {
            "NOT ENOUGH INFO": defaultdict(float),
            "SUPPORTS": defaultdict(float),
            "REFUTES": defaultdict(float)
        }

        # total proportions for each label
        label_proportions = defaultdict(int)

        # determine which feature computation to use (default unigram)
        self.tokenisation_type = tokenisation_type
        self.token_splitter = self.get_token_splitter(self.tokenisation_type)

        # loop over train set
        for example in train_set:
            claim = example['claim']

            if self.lowercase:
                claim = claim.lower()

            tokenized_claim = self.token_splitter(claim)
            label = example['label']
            label_proportions[label] +=1

            # loop over words in claim
            for token in tokenized_claim:
                # add token count
                label_to_word_freq_dict[label][token] += 1.0

        self.label_proportions = label_proportions
        self.labels = sorted(self.label_proportions.keys())

        # normalise by total label frequency
        for label in label_to_word_freq_dict.keys():
            label_proportion = label_proportions[label]
            for word in label_to_word_freq_dict[label].keys():
                label_to_word_freq_dict[label][word] = label_to_word_freq_dict[label][word] / label_proportion

        self.label_to_word_freq_dict = label_to_word_freq_dict

        # optional
        self.get_token_entropies()
        print("Done fitting Naive Bayes.")


    def get_token_entropies(self):
        """
        Computes entropy and probabiltiy distribution per token
        """
        # identify all tokens in the train set
        all_tokens = set()
        for label in self.labels:
            all_tokens.update(self.label_to_word_freq_dict[label].keys())

        # compute (normalised) probabilities for each label, given each token
        token2entropy = {}
        token2probabilities = {}

        for token in all_tokens:
            # probabilities: as many values as labels
            probabilities = [self.label_to_word_freq_dict[label][token] for label in self.labels]

            # normalise
            normaliser = sum(probabilities)
            probabilities = [x/normaliser for x in probabilities]
            assert np.abs(sum(probabilities) - 1.0) < 0.01

            # compute entropy
            entropy = -np.sum([p * np.log(p) for p in probabilities if p > 0.0])
            token2entropy[token] = entropy
            token2probabilities[token] = probabilities


        self.token2entropy = token2entropy
        self.token2probabilities = token2probabilities
        return token2entropy, token2probabilities

    def print_indicative_words(self, mode="positive"):
        """
        A function purely to play around with, for data exploration.
        Purpose: print out words strongly associated with one clss
        """
        print(self.labels)
        if mode == "positive":
            # prints the top 2000 tokens with highest probability for "SUPPORTS" (but only if seen at least 10 times)
            indicative_positive_tokens = [x for x in sorted(self.token2probabilities.items(), key=lambda x: -x[1][2])][:2000]
            for record in indicative_positive_tokens:
                if self.label_to_word_freq_dict["SUPPORTS"][record[0]]>10/self.label_proportions["SUPPORTS"]:
                    print(record)
        if mode == "negative":
            # prints the top 2000 tokens with highest probability for "REFUTES" (but only if seen at least 10 times)
            indicative_negative_tokens = [x for x in sorted(self.token2probabilities.items(), key=lambda x: -x[1][1])][:2000]
            for record in indicative_negative_tokens:
                if self.label_to_word_freq_dict["REFUTES"][record[0]]>10/self.label_proportions["REFUTES"]:
                    print(record)
        if mode == "neutral":
            # prints the top 2000 tokens with highest probability for "NOT ENOUGH INFO" (but only if seen at least 5 times)
            indicative_neutral_tokens = [x for x in sorted(self.token2probabilities.items(), key=lambda x: -x[1][0])][:2000]
            for record in indicative_neutral_tokens:
                if self.label_to_word_freq_dict["NOT ENOUGH INFO"][record[0]]>5/self.label_proportions["NOT ENOUGH INFO"]:
                    print(record)
        print(self.labels)

    def get_token_splitter(self, type="unigram"):
        """
        Returns a "tokenisation" function, but potentially also for bigrams,
        or for both unigrams and bigrams.
        """
        if type=="unigram":
            return lambda s: tokenize(s)
        elif type=="bigram":
            return bigram_splitter
        elif type=="both":
            # concatenation of both unigrams and bigrams
            return lambda s: tokenize(s) + bigram_splitter(s)
        else:
            return lambda s: []



    def predict_single_example(self, example):
        """
        Makes a Naive Bayes Prediction on a single given example.
        THis assumes a balanced test set distribution.
        """

        # will hold log probability for each label.
        conditional_probability_per_label = defaultdict(float)

        # featurise the given claim
        claim = example['claim']
        if self.lowercase:
            claim = claim.lower()
        tokenized_claim = self.token_splitter(claim)


        # loop over words in the claim
        for token in tokenized_claim:

            # get log probability for each label
            for label in self.labels:

                # look up probability from previously computed counts
                token_probability = self.label_to_word_freq_dict[label][token]

                if token_probability: # can be zero for OOV words.
                    conditional_probability_per_label[label] += np.log(token_probability)

        # rule out labels for which there is overall 0 probability
        conditional_probability_per_label = {k:v for k,v in conditional_probability_per_label.items() if v != 0.0}

        if conditional_probability_per_label:
            # default case: dictionary not empty
            prediction = max(conditional_probability_per_label.items(), key=lambda x: x[1])[0]
        else:
            # if no word has been observed for any of the labels --> predict random label
            prediction = random.choice(self.labels)

        return prediction


    def batch_predict(self, dataset):
        """
        Generates a dictionary: id -> prediction for each example in the given dataset
        """
        id2prediction = {}
        for example in dataset:
            prediction = self.predict_single_example(example)
            id2prediction[example['id']] = prediction
        return id2prediction






if __name__ == "__main__":

    # load some data (splitting off dev data from original train set)
    train, dev = load_split_trainset(dev_size=1000)

    # train a naive bayes model on Bag-of-Words features
    nb = NaiveBayesBow(train, tokenisation_type="unigram")

    # [for fun]: print out most frequent tokens for one of the classes
    print(sorted(nb.label_to_word_freq_dict["SUPPORTS"].items(), key= lambda x: -x[1] )[:10])


    # [for fun]: print out words most strongly associated with one class, vs. the others
    #nb.print_indicative_words("positive")
    nb.print_indicative_words("negative")
    #nb.print_indicative_words("neutral")

    # generate predictions for dev set
    dev_predictions = nb.batch_predict(dev)

    # evaluate the model on dev
    dev_labels = {x['id']: x['label'] for x in dev}
    calculate_accuracy(dev_labels, dev_predictions)
