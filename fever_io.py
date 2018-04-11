import json
import random


def get_label_set():
    label_set = {"SUPPORTS","REFUTES","NOT ENOUGH INFO"}
    return label_set


def load_split_trainset(dev_size:int):
    """
    Loads the full training set, splits it into preliminary train and dev set.
    This preliminary dev set is balanced.
    dev_size: size of dev set.
    """

    # load fever training data
    full_train = load_fever_train()

    positives = []
    negatives = []
    neutrals = []

    # sort dataset according to label.
    for example in full_train:
        example['id']
        label = example['label']
        if label == "SUPPORTS":
            positives.append(example)
        elif label == "REFUTES":
            negatives.append(example)
        elif label == "NOT ENOUGH INFO":
            neutrals.append(example)
        else:
            raise AssertionError("Bad label!", label)

    # shuffle examples for each label.
    random.seed(42)
    random.shuffle(positives)
    random.shuffle(negatives)
    random.shuffle(neutrals)

    # split off a preliminary dev set, balanced across each of the three classes
    size = int(dev_size/3)
    preliminary_dev_set = positives[:size] + negatives[:size] + neutrals[:size]

    # the remaining data will be the new training data
    train_set = positives[size:] + negatives[size:] + neutrals[size:]

    # shuffle order of examples
    random.shuffle(preliminary_dev_set)
    random.shuffle(train_set)

    return train_set, preliminary_dev_set


def load_fever_train(path="data/train.jsonl", howmany=999999):
    """
    Reads the Fever Training set, returns list of examples.
    howmany: how many examples to load. Useful for debugging.
    """
    data = []
    with open(path, 'r') as openfile:
        for iline, line in enumerate(openfile.readlines()):
            data.append(json.loads(line))
            if iline+1 >= howmany:
                break
    return data






if __name__ == "__main__":

    # load fever training data
    fever_data = load_fever_train(howmany=20)
    print(len(fever_data))

    # load train and split-off dev set of size 9999.
    train, dev = load_split_trainset(9999)
    print(len(train))
    print(len(dev))


    for sample in train[:3]:
        print(sample)

    # s = Sample(train[0])
    # print(s.__dict__)
