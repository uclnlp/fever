# FEVER Shared Task 2018
The First Workshop on Fact Extraction and Verification

To reproduce our FEVER shared task results:

## Initial steps
1. switch to the takuma-dev branch
2. clone this fork of the jack repo: https://github.com/takuma-ynd/jack
3. `cd jack` and download required data by `bash ./data/GloVe/download.sh`
4. create a data directory within the fever repo and download the FEVER [train.jsonl](https://s3-eu-west-1.amazonaws.com/fever.public/train.jsonl), [dev.jsonl](https://s3-eu-west-1.amazonaws.com/fever.public/shared_task_dev.jsonl) and [shared_task_test.jsonl](https://s3-eu-west-1.amazonaws.com/fever.public/shared_task_test.jsonl) datasets
5. download [wiki-data](https://s3-eu-west-1.amazonaws.com/fever.public/wiki-pages.zip) and place it under data/wiki-pages/ (Note: path to a json file will be `data/wiki-pages/wiki-pages/wiki-xxx.jsonl`)
6. run `bash setup.sh` to automatically set up the directory path

## Reproduce our result
1. run `bash download.sh` to get our trained model and required data
2. `python3 pipeline.py --config configs/submission_config.json --model [arbitrary name]`

## Train a model with new data
1. run `python3 doc_ir_model.py` to create a document index and retrieval model
2. run `python3 line_ir_model.py` to create a line index and retrieval model
3. create `configs/config.json` and reflect your directory structure (you can refer to `submission_config.json`)
4. `python3 pipeline.py --config configs/config.json --model [arbitrary name]`

## Configuration files
Configuration files can have a parent, which is specified by `parent_config` attribute.
attributes that are not specified in the child config file are inherited from the parent.
Usually you do not need to modify the parent config.

Note that currently a child cannot have its child. (Having a grand-parent or grand-child is not supported)

#### ir
Information Retrieval module.

#### convert
Data format conversion module (to let jack handle our data).

#### train_rte
Recognizing Textual Entailment module (training).

#### inference_rte
Recognizing Textual Entailment module (inference).

#### aggregator
Aggregation module

#### rerank
Reranking module

#### score
Evaluation module

## Original Paper
[UCL Machine Reading Group:
Four Factor Framework For Fact Finding (HexaF)](http://aclweb.org/anthology/W18-5515)
