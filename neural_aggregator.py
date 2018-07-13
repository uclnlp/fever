import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
sys.path.append("/Users/yoneda/workspace/UCL/fever/fever/")
from fever_io import read_jsonl, save_jsonl


class Net(nn.Module):
    def __init__(self, layers=[15,10,5]):
        super(Net, self).__init__()
        assert len(layers) == 3, "currently, only supports 3 layer MLP"
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(layers[0], layers[1])
        self.fc2 = nn.Linear(layers[1], layers[2])
        self.fc3 = nn.Linear(layers[2], 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class PredictedLabelsDataset(Dataset):
    """Predicted Labels dataset."""

    def __init__(self, jsonl_file):
        """
        """
        self.instances = read_jsonl(jsonl_file)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        if len(self.instances[idx]["predicted_labels"]) < 5:
            idx = 0
        return (create_target(self.instances[idx]["label"]),
                create_input(self.instances[idx]["predicted_labels"],
                             self.instances[idx]["scores"]))
        # return (self.instances[idx]["label"], self.instances[idx]["predicted_labels"]) #, self.instances[idx]["scores"])


def create_input(predicted_labels, scores):
    pred_labels = [label2idx[pred_label] for pred_label in predicted_labels]
    # print("pred_labels", pred_labels)

    one_hot = eye[
        pred_labels, :]  # equivalent to embedding_lookup(pred_labels, eye)
    # print("one_hot", one_hot)
    # print(np.multiply(one_hot, np.expand_dims(scores, axis=1)))

    # np_out = np.mean(np.multiply(one_hot, np.expand_dims(scores, axis=1)), axis=0)
    np_out = np.reshape(
        np.multiply(one_hot, np.expand_dims(scores, axis=1)), (-1))
    return np_out


def create_target(label):
    label_idx = label2idx[label]
    return label_idx


# use primodial aggregate preds function
def aggregate_preds(predictions):
    vote = dict()
    for rank, pred in enumerate(predictions):
        if pred not in vote:
            vote[pred] = 1
        else:
            vote[pred] += 1

    if supports in vote and refutes in vote:
        for pred in predictions:
            if pred in [supports, refutes]:
                final_verdict = pred
                break
    elif supports in vote:
        final_verdict = supports
    elif refutes in vote:
        final_verdict = refutes
    else:
        final_verdict = nei
    return final_verdict


def simple_test(dev_dataloader):
    # test for prediction
    neural_hit = 0
    heuristic_hit = 0
    with torch.no_grad():
        for i, (target, input) in enumerate(dev_dataloader):
            neural_pred = net(input.float())
            _, pred_labels = torch.max(neural_pred, 1)
            neural_hit += torch.sum(pred_labels == target)
    print("neural:", int(neural_hit) / len(dev_dataloader.dataset))

    for i, instance in enumerate(dev_dataloader.dataset.instances):
        heuristic_pred_label = aggregate_preds(instance["predicted_labels"])
        if heuristic_pred_label == instance["label"]:
            heuristic_hit += 1
    print("heuristic:", heuristic_hit / len(dev_dataloader.dataset.instances))


def predict(dev_dataloader):
    results = list()
    with torch.no_grad():
        for i, (labels, input) in enumerate(dev_dataloader):
            neural_preds = net(input.float())
            _, pred_labels = torch.max(neural_preds, 1)

            for label, neural_pred in zip(labels, pred_labels):
                results.append({
                    "actual": idx2label[int(label)],
                    "predicted": idx2label[int(neural_pred)]
                })

    return results


label2idx = {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": 2}
idx2label = {idx: label for label, idx in label2idx.items()}
supports = idx2label[0]
refutes = idx2label[1]
nei = idx2label[2]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--predicted_labels", required=True)
    parser.add_argument("--layers", nargs="+", required=True, help="<required> specify each width of layers (currently, should be 3 layers)")
    args = parser.parse_args()

    eye = np.eye(3)
    # data: prepend_title_linum
    train_set = PredictedLabelsDataset(args.train)
    dev_set = PredictedLabelsDataset(args.dev)
    train_dataloader = DataLoader(
        train_set, batch_size=64, shuffle=True, num_workers=4)
    dev_dataloader = DataLoader(
        dev_set, batch_size=64, shuffle=False, num_workers=4)

    net = Net(layers=[int(width) for width in args.layers])
    print("----Neural Aggregator Architecture----")
    print(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())
    for epoch in range(args.epochs):  # loop over the dataset multiple times
        print("epoch:", epoch)
        running_loss = 0.0
        for i, (labels, inputs) in enumerate(train_dataloader):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs.float())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 5000 == 4999:    # print every 5000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 5000))
                running_loss = 0.0

    print('Finished Training')

    simple_test(dev_dataloader)
    results = predict(dev_dataloader)
    save_jsonl(results, args.predicted_labels)
