# fever
FEVER Workshop Shared-Task


To reproduce our FEVER shared task results:

1. switch to the takuma-dev branch
2. clone this fork of the jack repo: https://github.com/takuma-ynd/jack
3. create a data directory within the fever repo and download the FEVER train.jsonl and dev.jsonl datasets
4. run doc_ir_model.py to create a document index and retrieval model
5. run line_ir_model.py to create a line index and retrieval model
6. edit configs/config.json to reflect your directory structure
7. python3 pipeline.py --config configs/config.json --model [arbitrary name]
