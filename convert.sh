mkdir -p data/converted
mkdir -p data/wiki-pages
ln -s ../../fever-baselines/data/fever data/fever
ln -s ../../fever-baselines/data/fever-data data/fever-data
ln -s ../../../fever-baselines/data/wiki-pages data/wiki-pages/wiki-pages
python3 converter.py data/fever/train.ns.rand.jsonl data/converted/train.ns.rand.converted.jsonl
python3 converter.py data/fever/dev.ns.rand.jsonl data/converted/dev.ns.rand.converted.jsonl
