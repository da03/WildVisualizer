# WildVisualizer

WildVisualizer is a tool for visualizing and exploring conversation datasets using Elasticsearch and UMAP embeddings.

## Dependencies

- Elasticsearch
- UMAP-learn
- Datasets
- OpenAI
- tiktoken
- TensorFlow
- Keras
- Flask
- Gunicorn

For Elasticsearch, follow the official installation and setup guide.

You can install most Python dependencies using:

```
pip install flask gunicorn numpy scikit-learn datasets umap-learn openai tqdm tiktoken elasticsearch tensorflow keras
```

## Setup and Deployment

### Build Elasticsearch Indices

First, we need to build Elasticsearch indices to enable efficient searching:

```
python create_es_wildchat.py
python create_es_lmsys.py
```

### Precompute Embeddings and Train UMAP Models

Next, we embed conversations and train parametric UMAP models to project them down to 2D. Note that we learn one projection per language:

```
python precompute_embeddings_and_umap.py
```

This step caches the embeddings for conversations sampled for training UMAP models. It also gzips a subset of conversation embeddings to enable efficient loading of the embedding page.

Note: Ensure that static gzip is enabled in nginx so that gzipped files are transmitted instead of uncompressed files.

### Build Elasticsearch Index for 2D Visualization

To ensure that all matching results in the displayed points are highlighted when users search in the 2D visualization, we build specific Elasticsearch indices for these subsets:

```
python create_es_wildchat_subset.py
python create_es_lmsys_subset.py
```

This doesn't limit users to searching within these subsets in the embedding visualization page. If not enough matches are found, the search falls back to the full index.

### Deploy Web Server

Finally, deploy the Flask-based web server:

```
gunicorn -w 4 -b 127.0.0.1:9972 main:app
```

## Features

- Interactive search and filtering capabilities
- Language-specific 2D visualizations of conversation embeddings
- Responsive design for both desktop and mobile devices
- Efficient loading and rendering of large datasets


## Acknowledgements

This repo is built on top of MiniConf from [Hendrik Strobelt](http://twitter.com/hen_str) and [Sasha Rush](http://twitter.com/srush_nlp).
