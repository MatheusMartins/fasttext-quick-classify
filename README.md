fasttext-quick-classify is a collection of two scripts and an example .csv, to show case the usage of fasttext.

* * *

The `.bin` model from fast-text must also be downloaded - [here](https://fasttext.cc/docs/en/english-vectors.html) or [here](https://fasttext.cc/docs/en/crawl-vectors.html). 

Tested with `cc.en.300.bin` - [link](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz).

* * * 

- Required Python Packages: numpy, chromadb, fasttext

- `load_data.py` run to create chromadb

- `classify_data.py` run to get the most similar entries according to inputs