#!/bin/bash

wget http://tti-coin.jp/data/yoneda/fever/base+sampling2+evscores+rerank+train+dev+test-shared_test.ver0727_newaggr_submission.zip -O results/base+sampling2+evscores+rerank+train+dev+test-shared_test.ver0727_newaggr_submission.zip
wget http://tti-coin.jp/data/yoneda/fever/data.zip -O data.zip
unzip results/base+sampling2+evscores+rerank+train+dev+test-shared_test.ver0727_newaggr_submission.zip -d result
unzip data
