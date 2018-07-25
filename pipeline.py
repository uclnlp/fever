import collections
from copy import deepcopy
import logging
import datetime
import argparse
import json
import os
from contextlib import contextmanager
import subprocess
from config_parser import parse
from fever_io import read_jsonl, save_jsonl


root_dir = "/cluster/project2/mr/tyoneda/pipeline"
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
    options = list()
    options.extend(["--train_input", ir_config["train_input_file"]])
    options.extend(["--dev_input", ir_config["dev_input_file"]])
    options.extend(["--train_output", ir_config["train_output_file"]])
    options.extend(["--dev_output", ir_config["dev_output_file"]])
    options.extend(["--n_docs", str(ir_config["n_pages"])])
    options.extend(["--n_sents", str(ir_config["n_sentences"])])
    script = ["get_evidence.py"] + options
    __run_python(script, gpu=False)


def convert(config):
    options = list()
    options.append(config["train_input_file"])
    options.append(config["train_converted_file"])
    if config["prependtitle"]:
        options.append("--prependtitle")
    if config["prependlinum"]:
        options.append("--prependlinum")
    if config["use_ir_pred"]:
        options.append("--use_ir_pred")
    if config["n_sentences"]:
        options.extend(["--n_sentences", str(config["n_sentences"])])

    # train data
    if not os.path.exists(config["train_converted_file"]):
        script = ["converter.py"] + options
        __run_python(script, gpu=False, env={"PYTHONPATH": "."})
    else:
        logger.info("%s already exists. skipping conversion for train", config["train_converted_file"])

    # dev data
    if not os.path.exists(config["dev_converted_file"]):
        options[0] = config["dev_input_file"]
        options[1] = config["dev_converted_file"]
        script = ["converter.py"] + options + ["--test"]
        __run_python(script, gpu=False, env={"PYTHONPATH": "."})
    else:
        logger.info("%s already exists. skipping conversion for dev", config["dev_converted_file"])


def train_rte(config):
    os.chdir("../jack")
    options = list()
    options.append("with")
    options.append("config={}".format(config["jack_config_file"]))
    options.append("save_dir={}".format(config["save_dir"]))
    options.append("train={}".format(config["train_file"]))
    options.append("dev={}".format(config["dev_file"]))
    options.append("test={}".format(config["dev_file"]))
    if "load_dir" in config and config["load_dir"] != "":
        options.append("load_dir={}".format(config["load_dir"]))

    script = ["bin/jack-train.py"] + options
    __run_python(script, gpu=True, env={"PYTHONPATH": "."})
    os.chdir("../fever")


def inference_rte(config):
    os.chdir(os.path.join(root_dir, "jack"))
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

    # train data
    if not os.path.exists(config["train_predicted_labels_and_scores_file"]):
        script = ["../fever/jack_reader.py"] + options
        __run_python(script, gpu=True, env={"PYTHONPATH": "."})
    else:
        logger.info("skipping inference rte for train. %s exists", config["train_predicted_labels_and_scores_file"])

    # dev data
    if not os.path.exists(config["dev_predicted_labels_and_scores_file"]):
        options[0] = config["dev_input_file"]
        options[1] = config["dev_predicted_labels_and_scores_file"]
        script = ["../fever/jack_reader.py"] + options
        __run_python(script, gpu=True, env={"PYTHONPATH": "."})
    else:
        logger.info("skipping inference rte for dev. %s exists", config["dev_predicted_labels_and_scores_file"])

    # test data
    if not os.path.exists(config["test_predicted_labels_and_scores_file"]):
        options[0] = config["test_input_file"]
        options[1] = config["test_predicted_labels_and_scores_file"]
        script = ["../fever/jack_reader.py"] + options
        __run_python(script, gpu=True, env={"PYTHONPATH": "."})
    else:
        logger.info("skipping inference rte for test. %s exists", config["test_predicted_labels_and_scores_file"])

def neural_aggregator(config):
    os.chdir(os.path.join(root_dir, "fever"))
    options = list()
    options.extend(["--train", config["train_file"]])
    options.extend(["--dev", config["dev_file"]])
    options.extend(["--test", config["test_file"]])
    options.extend(["--epochs", str(config["epochs"])])
    options.extend(["--predicted_labels", config["predicted_labels_file"]])
    options.extend(["--test_predicted_labels", config["test_predicted_labels_file"]])
    options.extend(["--n_sentences", str(config["n_sentences"])])
    layers = [str(num) for num in config["layers"]]
    options.extend(["--layers"] + layers)
    if "sampling" in config and config["sampling"]:
        options.append("--sampling")
    if "ev_scores" in config and config["ev_scores"]:
        options.append("--ev_scores")

    script = ["neural_aggregator.py"] + options
    __run_python(script, gpu=False)


def score(config):
    os.chdir(os.path.join(root_dir, "fever-baselines"))
    options = list()
    options.extend(["--predicted_labels", config["predicted_labels_file"]])
    options.extend(["--predicted_evidence", config["predicted_evidence_file"]])
    options.extend(["--actual", config["actual_file"]])
    options.extend(["--score_file", config["score_file"]])
    options.extend(["--submission_file", config["submission_file"]])

    script = ["src/scripts/score.py"] + options
    __run_python(script, gpu=False, env={"PYTHONPATH": "src:../fever"})


    if "test" in config and config["test"]:
        options = list()
        options.extend(["--predicted_labels", config["test"]["predicted_labels_file"]])
        options.extend(["--predicted_evidence", config["test"]["predicted_evidence_file"]])
        options.extend(["--actual", config["test"]["actual_file"]])
        options.extend(["--score_file", config["test"]["score_file"]])
        options.extend(["--submission_file", config["test"]["submission_file"]])

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
            # "CUDA_VISIBLE_DEVICES": "0"
        })
        prep = python_gpu_prep

    with environ(env):
        script = prep + script
        logger.info("running: %s", script)
        ret = subprocess.run(script)
        if ret.returncode != 0:
            logger.info("returned: %s", ret)
            raise RuntimeError("shell returned non zero code.")


def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    now = datetime.datetime.now()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--model", default="{0:model_%Y%m%d%H%M%S}".format(now))
    parser.add_argument(
        "--overwrite", action="store_true")
    args = parser.parse_args()
    if os.path.exists(os.path.join("results", args.model, "org_config.json")) and not args.overwrite:
        logger.warning("overwriting the existing model due to --overwrite flag.")
        raise RuntimeError("you cannot overwrite the config. use different model name.")

    with open(args.config) as f:
        config = json.load(f)

    # load and save original config
    if "__variables" not in config:
        config["__variables"] = {}
    config["__variables"]["___model_name___"] = args.model
    model_dir = "results/{}".format(config["__variables"]["___model_name___"])

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    logger.info("model dir: %s", model_dir)
    save_config(config, path=os.path.join(model_dir, "org_config.json"))

    # load child config if specified
    if "parent_config" in config and config["parent_config"]:
        path = config["parent_config"]
        logger.info("loading parent config from {}".format(path))
        with open(path) as f:
            parent_config = json.load(f)
        save_config(parent_config, path=os.path.join(model_dir, "parent_config.json"))
        config = update(deepcopy(parent_config), config)

    config = parse(config)
    save_config(config, path=os.path.join(model_dir, "config.json"))

    # perform IR if file doesn't exist
    config_ir = config["ir"]
    logger.info("%s exists?: %s", config_ir["train_output_file"], os.path.exists(config_ir["train_output_file"]))
    logger.info("%s exists?: %s", config_ir["dev_output_file"], os.path.exists(config_ir["dev_output_file"]))
    if not (os.path.exists(config_ir["train_output_file"])
            and os.path.exists(config_ir["dev_output_file"])):
        ir(config["ir"])
    else:
        logger.info("skipping ir...")

    # convert format if file does not exist
    conf_convert = config["convert"]
    logger.info("%s exists?: %s", conf_convert["train_converted_file"], os.path.exists(conf_convert["train_converted_file"]))
    logger.info("%s exists?: %s", conf_convert["dev_converted_file"], os.path.exists(conf_convert["dev_converted_file"]))
    logger.info("%s exists?: %s", conf_convert["test_converted_file"], os.path.exists(conf_convert["test_converted_file"]))
    if not( os.path.exists(
            conf_convert["train_converted_file"]) and os.path.exists(
                conf_convert["dev_converted_file"]) and os.path.exists(
                    conf_convert["test_converted_file"])):
        convert(conf_convert)
    else:
        logger.info("skipping conversion...")

    # train rte model if file does not exist
    conf_train_rte = config["train_rte"]
    logger.info("%s exists?: %s", conf_train_rte["save_dir"], os.path.exists(conf_train_rte["save_dir"]))
    if not os.path.isdir(conf_train_rte["save_dir"]):
        train_rte(conf_train_rte)
    else:
        logger.info("skipping train rte...")

    # rte inference if file does not exist
    conf_inference = config["inference_rte"]
    logger.info("%s exists?: %s", conf_inference["train_predicted_labels_and_scores_file"], os.path.exists(conf_inference["train_predicted_labels_and_scores_file"]))
    logger.info("%s exists?: %s", conf_inference["dev_predicted_labels_and_scores_file"], os.path.exists(conf_inference["dev_predicted_labels_and_scores_file"]))
    logger.info("%s exists?: %s", conf_inference["test_predicted_labels_and_scores_file"], os.path.exists(conf_inference["test_predicted_labels_and_scores_file"]))
    if not os.path.exists(
            conf_inference["train_predicted_labels_and_scores_file"]) or not os.path.exists(conf_inference["dev_predicted_labels_and_scores_file"]) or not os.path.exists(conf_inference["test_predicted_labels_and_scores_file"]):
        inference_rte(config["inference_rte"])
    else:
        logger.info("skipping inference rte...")

    # aggregation if file not exists
    if not os.path.exists(config["aggregator"]["predicted_labels_file"]):
        neural_aggregator(config["aggregator"])
    else:
        logger.info("skipping aggregation...")

    # scoring
    if not os.path.exists(config["score"]["score_file"]):
        score(config["score"])
