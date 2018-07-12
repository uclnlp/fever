import datetime
import argparse
import json
import os
import subprocess
from config_parser import parse
from fever_io import read_jsonl, save_jsonl


def load_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def save_config(config, save_path):
    with open(save_path, "w") as f:
        json.dump(config, f)


def ir(ir_config):
    raise RuntimeError("this func is under construction!")
    train_target = "train_target={}".format(ir_config["train_target_file"])
    dev_target = "dev_target={}".format(ir_config["dev_target_file"])
    subprocess.run(["python3", "get_evidences.py"])


def convert(config):
    raise RuntimeError("this func is under construction!")
    options = list()
    options.append(config)
    if config["prependtitle"]:
        options.append("--prependtitle")
    if config["prependlinum"]:
        options.append("--prependlinum")
    if config["use_ir_pred"]:
        options.append("--use_ir_pred")
        options.append("--n_sentences {}".format(config["n_sentences"]))

    script = ["python3", "converter.py"] + options
    print(" ".join(script))
    subprocess.run(script)


def train_rte(config):
    raise RuntimeError("this func is under construction!")
    options = list()
    options = ["config='{}'".format(config["jack_config_file"])]
    options.append("--save_dir {}".format(config["save_dir"]))
    options.append("--train {}".format(config["train"]))
    options.append("--dev {}".format(config["dev"]))

    script = ["anaconda-python3-gpu", "bin/jack_train.py", "with"] + options
    subprocess.run(script)


def inference_rte(config):
    os.chdir("/home/tyoneda/pipeline/jack")
    options = list()
    options.append("--saved_reader {}".format(config["saved_reader"]))
    if config["prependlinum"]:
        options.append("--prependlinum")
    if config["prependtitle"]:
        options.append("--prependtitle")
    if config["n_sentences"]:
        options.append("--n_sentences")

    # train data
    options.append("--save_preds {}".format(config["train_predicted_labels_and_scores_file"]))
    script = ["anaconda-python3-gpu", "jack_reader.py"] + options
    print(script)
    subprocess.run(script)

    # dev data
    options[-1] = "--save_preds {}".format(config["dev_predicted_labels_and_scores_file"])
    print(script)
    script = ["anaconda-python3-gpu", "jack_reader.py"] + options
    subprocess.run(script)


def neural_aggregator(config):
    os.chdir("/home/tyoneda/pipeline/fever")
    options = list()
    options.append("--train_file {}".format(config["train_file"]))
    options.append("--dev_file {}".format(config["dev_file"]))
    options.append("--epochs {}".format(config["epochs"]))
    options.append("--predicted_labels {}".format(
        config["predicted_labels_file"]))
    script = ["python3", "neural_aggregator.py"] + options
    print(script)
    subprocess.run(script)


def score(config):
    os.chdir("/home/tyoneda/pipeline/fever-baslines")
    options = list()
    options.append("--predicted_labels {}".format(
        config["predicted_labels_file"]))
    options.append("--predicted_evidence {}".format(
        config["predicted_evidence_file"]))
    options.append("--actual {}".format(config["actual_file"]))
    options.append("--out_file {}".format(config["out_file"]))
    script = ["python3", "src/script/score.py"] + options
    print(script)
    subprocess.run(script)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    now = datetime.datetime.now()
    # load config
    config["__variables"]["___model_name___"] = "{0:model_%Y%m%d%H%M%S}".format(now)
    config = parse(config)
    model_dir = "results/{}".format(config["__variables"]["___model_name___"])
    if not os.path.exits(model_dir):
        os.mkdir(model_dir)

    save_config(config, path="result/{}/config.json".format(config["__variables"]["___model_name___"]))

    # perform IR if file doesn't exist
    if not (os.path.exists(config["ir"]["train_target_file"])
            and os.path.exists(config["ir"]["dev_target_file"])):
        ir(config["ir"])
    else:
        print("skipping ir...")

    # convert format if file does not exist
    if not os.path.exists(config["convert"]["train_converted_file"]) and os.path.exists(config["convert"]["dev_converted_file"]):
        convert(config["convert"])
    else:
        print("skipping conversion...")

    # train rte model if file does not exist
    if not os.path.isdir(config["train_rte"]["save_dir"]):
        train_rte(config["train_rte"])
    else:
        print("skipping train rte...")

    # rte inference if file does not exist
    if not os.path.exists(config["inference_rte"]["train_predicted_labels_and_scores_file"]):
        inference_rte(config["inference_rte"])
    else:
        print("skipping inference rte...")

    # aggregation if file not exists
    if not os.path.exists(config["aggregator"]["predicted_labels_file"]):
        neural_aggregator(config["aggregator"])
    else:
        print("skipping aggregation...")

    # scoring
    if not os.path.exists(config["score"]["score_file"]):
        score(config["score"])
