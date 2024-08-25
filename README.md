# WildVisualizer

WildVisualizer is a tool for visualizing and exploring conversation datasets using Elasticsearch and UMAP embeddings.


## Features

- Interactive search and filtering capabilities
- Language-specific 2D visualizations of conversation embeddings
- Responsive design for both desktop and mobile devices
- Efficient loading and rendering of large datasets


## Dependencies

- Elasticsearch
- UMAP-learn (Due to a bug in parametric UMAP when working with large datasets, we are using a customized version available at https://github.com/da03/umap)
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
pip install flask gunicorn numpy scikit-learn datasets git+https://github.com/da03/umap openai tqdm tiktoken elasticsearch tensorflow keras
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


## Important Files and Directories

- `main.py`: The main Flask application file
- `templates/`:
  - `index.html`: Template for the main search page
  - `embeddings.html`: Template for the 2D embedding visualization page
- `static/`:
  - `js/views/embeddings.js`: JavaScript file for the embedding visualization
  - `css/main.css`: Main CSS file for styling the application
- `create_es_wildchat.py`: Script to build Elasticsearch index for WildChat dataset
- `create_es_lmsys.py`: Script to build Elasticsearch index for LMSYSChat dataset
- `precompute_embeddings_and_umap.py`: Script to compute embeddings and train UMAP models
- `create_es_wildchat_subset.py`: Script to build Elasticsearch index for the visualization subset of WildChat
- `create_es_lmsys_subset.py`: Script to build Elasticsearch index for the visualization subset of LMSYSChat


## Acknowledgements

This repo is built on top of MiniConf from [Hendrik Strobelt](http://twitter.com/hen_str) and [Sasha Rush](http://twitter.com/srush_nlp).
