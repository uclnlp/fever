# FEVER Shared Task 2018
The First Workshop on Fact Extraction and Verification

To reproduce our FEVER shared task results:

## Initial steps
1. switch to the takuma-dev branch
2. run `bash initial_setup.sh` (This will download several files and take some time.)
3. move to `jack` directory and install dependencies according to the README. (i.e., `python3 -m pip install -e .[tf]`)
4. move to `fever-baselines` directory and install dependencies (i.e., `pip install -r requirements.txt`)

After step 2, `fever`, `jack`, `fever-baselines` directory should be at the same level (these should be in the same directory).

## Reproduce our result
`python3 pipeline.py --config configs/submission_config.json --model [arbitrary name]`

Output files will be generated under `results/[arbitrary name]`.
`submission.json` and `test_submission.json` correspond to the output for development and test set for each. 
(**Note that the score displayed after running this particular model is not valid, since we included development set for the training even though the score is calculated based on the development set.**)

## Train a model with new data

### Use the same wiki data 
1. create new configuration file `configs/config.json` and reflect your directory structure
2. run `python3 pipeline.py --config configs/config.json --model [arbitrary name]`

### Use a new/different wiki-pages data
1. remove index files in `data` directory
2. run `python3 doc_ir_model.py` to create a document index and retrieval model
3. run `python3 line_ir_model.py` to create a line index and retrieval model


## Configuration files
Configuration files can have a parent, which is specified by `parent_config` attribute.

Attributes that are not specified in the child config file are inherited from the parent.
Usually you do not need to modify the parent config.

**Note that currently a child cannot have its child. (Having a grand-parent or grand-child is not supported)**

### ir
Information Retrieval module.

### convert
Data format conversion module (to let jack handle our data).

### train_rte
Recognizing Textual Entailment module (training).

### inference_rte
Recognizing Textual Entailment module (inference).

### aggregator
Aggregation module

### rerank
Reranking module

### score
Evaluation module

## Original Paper
[UCL Machine Reading Group:
Four Factor Framework For Fact Finding (HexaF)](http://aclweb.org/anthology/W18-5515)
