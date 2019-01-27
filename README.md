# FEVER Shared Task 2018
The First Workshop on Fact Extraction and Verification

To reproduce our FEVER shared task results:

1. switch to the takuma-dev branch
2. clone this fork of the jack repo: https://github.com/takuma-ynd/jack
3. `cd jack` and download required data by `bash ./data/GloVe/download.sh`
4. create a data directory within the fever repo and download the FEVER [train.jsonl](https://s3-eu-west-1.amazonaws.com/fever.public/train.jsonl), [dev.jsonl](https://s3-eu-west-1.amazonaws.com/fever.public/shared_task_dev.jsonl) and [test.jsonl](https://s3-eu-west-1.amazonaws.com/fever.public/shared_task_test.jsonl) datasets
5. download [wiki-data](https://s3-eu-west-1.amazonaws.com/fever.public/wiki-pages.zip) and place it under data/wiki-pages/ (Path to a json file will be `data/wiki-pages/wiki-pages/wiki-xxx.jsonl`)
6. run `bash download.sh` to get our trained model and other data
7. run `doc_ir_model.py` to create a document index and retrieval model
8. run `line_ir_model.py` to create a line index and retrieval model
9. create `configs/config.json` and reflect your directory structure (you can refer to `submission_config.json`)
10. `python3 pipeline.py --config configs/config.json --model [arbitrary name]`
