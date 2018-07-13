import datetime
import argparse
import json
import os
from contextlib import contextmanager
import subprocess
from config_parser import parse
from fever_io import read_jsonl, save_jsonl


@contextmanager
def environ(env):
    original_environ_dict = os.environ.copy()
    os.environ.update(env)
    yield
    os.environ.clear()
    os.environ.update(original_environ_dict)


def load_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def save_config(config, path):
    with open(path, "w") as f:
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

    script = " ".join(["anaconda-python3-gpu", "bin/jack_train.py", "with"] +
                      options)
    subprocess.run(script)


def inference_rte(config):
    os.chdir("/home/tyoneda/pipeline/jack")
    options = list()
    options.append(config["train_input_file"])  # input file
    options.append(config["train_predicted_labels_and_scores_file"])
    options.extend(["--saved_reader", config["saved_reader"]])
    if config["prependlinum"]:
        options.append("--prependlinum")
    if config["prependtitle"]:
        options.append("--prependtitle")
    if config["n_sentences"]:
        options.extend(["--n_sentences", str(config["n_sentences"])])

    # print(["python3", "hello_world.py"])
    # subprocess.run(["python3", "hello_world.py"])
    # quit()

    # train data
    script = ["../fever/jack_reader.py"] + options
    __run_python(script, gpu=True, env={"PYTHONPATH": "."})

    # dev data
    options[0] = config["dev_input_file"]
    options[1] = config["dev_predicted_labels_and_scores_file"]
    script = ["../fever/jack_reader.py"] + options
    __run_python(script, gpu=True, env={"PYTHONPATH": "."})


def neural_aggregator(config):
    os.chdir("/home/tyoneda/pipeline/fever")
    options = list()
    options.extend(["--train", config["train_file"]])
    options.extend(["--dev", config["dev_file"]])
    options.extend(["--epochs", str(config["epochs"])])
    options.extend(["--predicted_labels", config["predicted_labels_file"]])
    options.extend(["--n_sentences", str(config["n_sentences"])])
    layers = [str(num) for num in config["layers"]]
    options.extend(["--layers"] + layers)

    script = ["neural_aggregator.py"] + options
    __run_python(script, gpu=False)


def score(config):
    os.chdir("/home/tyoneda/pipeline/fever-baselines")
    options = list()
    options.extend(["--predicted_labels", config["predicted_labels_file"]])
    options.extend(["--predicted_evidence", config["predicted_evidence_file"]])
    options.extend(["--actual", config["actual_file"]])
    options.extend(["--score_file", config["score_file"]])

    script = ["src/scripts/score.py"] + options
    __run_python(script, gpu=False, env={"PYTHONPATH": "src:../fever"})


def __run_python(script, gpu=False, env=dict()):
    LD_LIBRARY_PATH = "/share/apps/cuda-9.0/lib64:/share/apps/python-3.6.3-shared/lib:/share/apps/libc6_2.23/lib/x86_64-linux-gnu:/share/apps/libc6_2.23/lib64:/share/apps/gcc-6.2.0/lib64:/share/apps/gcc-6.2.0/lib"
    python_gpu_prep = [
        "/share/apps/libc6_2.23/lib/x86_64-linux-gnu/ld-2.23.so",
        "/home/tyoneda/anaconda3/bin/python3"
    ]
    prep = ["/home/tyoneda/anaconda3/bin/python3"]
    if gpu:
        env.update({
            "LD_LIBRARY_PATH": LD_LIBRARY_PATH,
            "CUDA_VISIBLE_DEVICES": "0"
        })
        prep = python_gpu_prep

    with environ(env):
        script = prep + script
        print(script)
        ret = subprocess.run(script)
        if ret.returncode != 0:
            print(ret)
            raise RuntimeError("shell returned non zero code.")


if __name__ == '__main__':
    now = datetime.datetime.now()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--model", default="{0:model_%Y%m%d%H%M%S}".format(now))
    args = parser.parse_args()
    if os.path.exists(os.path.join("results", args.model, "org_config.json")):
        raise RuntimeError("you cannot overwrite the config. use different model name.")

    with open(args.config) as f:
        config = json.load(f)

    # load config
    config["__variables"]["___model_name___"] = args.model
    model_dir = "results/{}".format(config["__variables"]["___model_name___"])

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    save_config(config, path=os.path.join(model_dir, "org_config.json"))
    config = parse(config)
    save_config(config, path=os.path.join(model_dir, "config.json"))

    # perform IR if file doesn't exist
    if not (os.path.exists(config["ir"]["train_target_file"])
            and os.path.exists(config["ir"]["dev_target_file"])):
        ir(config["ir"])
    else:
        print("skipping ir...")

    # convert format if file does not exist
    if not os.path.exists(
            config["convert"]["train_converted_file"]) and os.path.exists(
                config["convert"]["dev_converted_file"]):
        convert(config["convert"])
    else:
        print("skipping conversion...")

    # train rte model if file does not exist
    if not os.path.isdir(config["train_rte"]["save_dir"]):
        train_rte(config["train_rte"])
    else:
        print("skipping train rte...")

    # rte inference if file does not exist
    if not os.path.exists(
            config["inference_rte"]["train_predicted_labels_and_scores_file"]):
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
