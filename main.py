# pylint: disable=global-statement,redefined-outer-name
import argparse
import csv
import glob
import keras
import json
import random
import os
from markupsafe import Markup, escape
import sqlite3
from openai import OpenAI
import numpy as np
import joblib
#from umap.parametric_umap import load_ParametricUMAP
import tiktoken
#from sklearn.decomposition import PCA
from umap.parametric_umap import ParametricUMAP
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


indices = ['wildchat', 'lmsyschat']
supported_fields = {'wildchat': ['dataset', 'toxic', 'redacted', 'model', 'hashed_ip', 'language', 'country', 'state', 'min_turns', 'conversation_id'],
                        'lmsyschat': ['dataset', 'toxic', 'redacted', 'model', 'language', 'min_turns', 'conversation_id']}
def build_query_for_index(index_name, filters, contains, from_, size_):
    # Build Elasticsearch query
    must_clauses = []
    if contains:
        must_clauses.append({
            "nested": {
                "path": "conversation",
                "query": {
                    "match_phrase": {
                        "conversation.content": contains
                    }
                }
            }
        })
    if filters['toxic']:
        if index_name == 'wildchat':
            must_clauses.append({"term": {"toxic": filters['toxic'] == 'true'}})
        else:
            must_clauses.append({
            "nested": {
                "path": "openai_moderation",
                "query": {
                    "term": {"openai_moderation.flagged": filters['toxic'] == 'true'}
                }
            }
        })
    if filters['redacted']:
        must_clauses.append({"term": {"redacted": filters['redacted'] == 'true'}})
    if filters['model']:
        must_clauses.append({"term": {"model": filters['model']}})
    if filters['hashed_ip']:
        if index_name == 'wildchat':
            must_clauses.append({"term": {"hashed_ip": filters['hashed_ip']}})
    if filters['language']:
        must_clauses.append({"term": {"language": filters['language'].title()}})
    if filters['country']:
        if index_name == 'wildchat':
            must_clauses.append({"term": {"country": filters['country']}})
    if filters['state']:
        if index_name == 'wildchat':
            must_clauses.append({"term": {"state": filters['state']}})
    if filters['min_turns']:
        must_clauses.append({"range": {"turn": {"gte": int(filters['min_turns'])}}})
    if filters['conversation_id']:
        if index_name == 'wildchat':
            must_clauses.append({"nested": {"path": "conversation", "query": {"term": {"conversation.turn_identifier": filters['conversation_id']}}}})
        else:
            must_clauses.append({"term": {"conversation_id": filters['conversation_id']}})


    search_query = {
        "query": {
            "bool": {
                "must": must_clauses if must_clauses else {"match_all": {}}
            }
        },
        "from": from_,
        "size": size_
    }

    if must_clauses:
        any_filters = True
    else:
        any_filters = False
    return search_query, any_filters

def nl2br(value):
    escaped_value = escape(value)
    return Markup(escaped_value.replace('\n', Markup('<br>')))

import yaml
from flask import Flask, jsonify, redirect, render_template, send_from_directory, request, url_for, send_file, abort, after_this_request
from flask_frozen import Freezer
from flaskext.markdown import Markdown
from elasticsearch import Elasticsearch, helpers

site_data = {}
by_uid = {}

es = Elasticsearch('https://localhost:9200', basic_auth=('elastic', os.getenv('ES_PASSWD')), ssl_assert_fingerprint=os.getenv('ES_FINGERPRINT'))

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
tokenizer = tiktoken.get_encoding('cl100k_base')

# Load the PCA model

embedding_projectors = {}
for folder in glob.glob(os.path.join('umap_model', '*')):
   #scaler_path = os.path.join(folder, 'scaler.pkl')
   umap_path = folder
   #if os.path.exists(scaler_path):
   if os.path.exists(umap_path):
       language = os.path.basename(folder)
       #scaler = joblib.load(scaler_path)
       try:
           umap_encoder = keras.models.load_model(os.path.join(umap_path, "encoder.keras"))
           #umap = load_ParametricUMAP(umap_path)
           #embedding_projectors[language] = {'scaler': scaler, 'umap': umap}
           embedding_projectors[language] = umap_encoder
       except Exception as e:
           print (e)
print (embedding_projectors.keys())

def create_database(db_name):
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS cache
                 (key TEXT PRIMARY KEY, prompt TEXT, embedding TEXT)''')
    conn.commit()
    conn.close()

#create_database('embeddings_cache.db')
#create_database('umap_cache.db')

def insert_or_update(db_name, key, prompt, embedding):
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    c.execute('''INSERT OR REPLACE INTO cache
                 (key, prompt, embedding) VALUES (?, ?, ?)''', 
                 (key, prompt, json.dumps(embedding)))
    conn.commit()
    conn.close()

def retrieve(db_name, key):
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    c.execute("SELECT embedding FROM cache WHERE key=?", (key,))
    result = c.fetchone()
    conn.close()
    if result:
        return True, json.loads(result[0])
    else:
        return False, None


def get_embedding_with_cache(database_name, conversation_id, prompt, model='text-embedding-3-small'):
    key = conversation_id
    hit, embedding = retrieve(database_name, key)
    if not hit:
        tokens = tokenizer.encode(prompt, disallowed_special=())
        #if len(tokens) > 8192:
        #    tokens = tokens[:8192]
        #    prompt = tokenizer.decode(tokens)
        if len(tokens) > 8100:
            tokens = tokens[:8100]
            prompt = tokenizer.decode(tokens)
        embedding = client.embeddings.create(input=[prompt], model=model).data[0].embedding
        insert_or_update(database_name, key, json.dumps(prompt), embedding)
    #else:
    #    print('Cache hit for embedding')
    return embedding


def main(site_data_path):
    global site_data, extra_files
    extra_files = ["README.md"]
    # Load all for your sitedata one time.
    for f in glob.glob(site_data_path + "/*"):
        extra_files.append(f)
        try:
            name, typ = f.split("/")[-1].split(".")
        except Exception as e:
            continue
        if typ == "json":
            site_data[name] = json.load(open(f))
        elif typ in {"csv", "tsv"}:
            site_data[name] = list(csv.DictReader(open(f)))
        elif typ == "yml":
            site_data[name] = yaml.load(open(f).read(), Loader=yaml.SafeLoader)

    for typ in ["papers", "speakers", "workshops"]:
        by_uid[typ] = {}
        for p in site_data[typ]:
            by_uid[typ][p["UID"]] = p

    print("Data Successfully Loaded")
    return extra_files

extra_files = main('sitedata')
# ------------- SERVER CODE -------------------->

app = Flask(__name__)
app.jinja_env.add_extension('jinja2.ext.do')
app.jinja_env.filters['nl2br'] = nl2br
app.config.from_object(__name__)
freezer = Freezer(app)
markdown = Markdown(app)


# MAIN PAGES
def _data():
    data = {}
    data["config"] = site_data["config"]
    return data

@app.route("/favicon.ico")
def favicon():
    return send_from_directory('sitedata', "favicon.ico")

# TOP LEVEL PAGES
@app.route("/")
def index():
    data = _data()
    contains = request.args.get('contains', '')
    page = int(request.args.get('page', 1))

    # Construct the Elasticsearch query
    filters = {
        "dataset": request.args.get('dataset', ''),
        "toxic": request.args.get('toxic', ''),
        "redacted": request.args.get('redacted', ''),
        "model": request.args.get('model', ''),
        "hashed_ip": request.args.get('hashed_ip', ''),
        "language": request.args.get('language', ''),
        "country": request.args.get('country', ''),
        "state": request.args.get('state', ''),
        "min_turns": request.args.get('min_turns', ''),
        "conversation_id": request.args.get('conversation_id', '')
    }


    disabled_datasets = []
    for dataset in indices:
        for field in filters:
            if filters[field]:
                if field not in supported_fields[dataset]:
                    disabled_datasets.append(dataset)
    indices_to_search = []
    if (filters['dataset'] == '' or filters['dataset'] == 'wildchat') and 'wildchat' not in disabled_datasets:
        indices_to_search.append('wildchat')
    if (filters['dataset'] == '' or filters['dataset'] == 'lmsyschat') and 'lmsyschat' not in disabled_datasets:
        indices_to_search.append('lmsyschat')
    size = max(30 // len(indices_to_search), 1)
    from_ = (page - 1) * size
    if from_ >= 10000:
        return render_template("error.html", message="You cannot navigate beyond the 10,000th result. Please refine your search by going to earlier pages.")
    if from_+size > 10000:
        size_ = 10000 - from_
    else:
        size_ = size

    any_filters = False
    if 'dataset' in filters and filters['dataset'] != '':
        any_filters = True
    conversations = []
    total = 0
    assert len(indices_to_search) > 0
    for index_name in indices_to_search:
        # Execute search query
        search_query, any_filters_ = build_query_for_index(index_name, filters, contains, from_, size_)
        any_filters = any_filters or any_filters_
        response = es.search(index=index_name, body=search_query)
        conversations_raw = [hit['_source'] for hit in response['hits']['hits']]

        for conversation_raw in conversations_raw:
            conversation = {}
            conversation['dataset'] = index_name
            for key in ['timestamp', 'country', 'state', 'hashed_ip', 'model', 'toxic', 'redacted', 'conversation', 'conversation_id']:
                if key in conversation_raw:
                    conversation[key] = conversation_raw[key]
            if index_name == 'wildchat':
                conversation['conversation_id'] = conversation_raw['conversation'][0]['turn_identifier']
            if index_name == 'lmsyschat':
                conversation['toxic'] = any([item['flagged'] for item in conversation_raw['openai_moderation']])
            conversations.append(conversation)
        total = max(total, response['hits']['total']['value'])
    #total_pages = (total // size) + 1
    total_pages = (total + size - 1) // size
    random.seed(1234)
    random.shuffle(conversations)

    # Pagination logic
    pages = []
    if total_pages > 1:
        if page > 3:
            pages.append(1)
            if page > 4:
                pages.append('...')
        pages.extend(range(max(1, page - 2), min(total_pages + 1, page + 3)))
        if page < total_pages - 3:
            if page < total_pages - 4:
                pages.append('...')
            pages.append(total_pages)
    #import pdb; pdb.set_trace()
    data.update({
        "conversations": conversations,
        "contains": contains,
        "page": page,
        "pages": pages,
        "total": total,
        "filters": filters,
        "any_filters": any_filters
    })
    return render_template("index.html", **data)

@app.route('/search_embeddings', methods=['POST'])
def search_embeddings():
    filters = request.json
    search_expansion_limit = filters['search_expansion_limit']
    del filters['search_expansion_limit']
    if search_expansion_limit == '':
        search_expansion_limit = '100'
    search_expansion_limit = int(search_expansion_limit)
    search_expansion_limit = max(0, min(search_expansion_limit, 2000))

    contains = filters['contains']
    del filters['contains']
    visualization_language = filters['visualization_language']
    del filters['visualization_language']


    #scaler = embedding_projectors[language]['scaler']
    #umap = embedding_projectors[language]['umap']
    umap_encoder = embedding_projectors[visualization_language]
    #print (filters)
    disabled_datasets = []
    for dataset in indices:
        for field in filters:
            if filters[field]:
                if field not in supported_fields[dataset]:
                    disabled_datasets.append(dataset)
    indices_to_search = []
    if (filters['dataset'] == '' or filters['dataset'] == 'wildchat') and 'wildchat' not in disabled_datasets:
        indices_to_search.append('wildchat')
    if (filters['dataset'] == '' or filters['dataset'] == 'lmsyschat') and 'lmsyschat' not in disabled_datasets:
        indices_to_search.append('lmsyschat')
    any_filters = False
    #import pdb; pdb.set_trace()
    for index_name in indices_to_search:
        # Execute search query
        search_query, any_filters_ = build_query_for_index(index_name, filters, contains, 0, 10000)
        any_filters = any_filters or any_filters_
    conversations = []
    if any_filters:
        if (('language' not in filters) or (not filters['language'])) and (visualization_language != 'all'):
            filters['language'] = visualization_language
        conversation_ids = set([])
        for index_name in indices_to_search:
            # Execute search query
            search_query, any_filters_ = build_query_for_index(index_name, filters, contains, 0, 10000)
            response = es.search(index=index_name + '_subset_' + visualization_language, body=search_query)
            conversations_raw = [hit['_source'] for hit in response['hits']['hits']]

            for conversation_raw in conversations_raw:
                conversation = {}
                conversation['dataset'] = index_name
                for key in ['conversation', 'conversation_id']:
                    if key in conversation_raw:
                        conversation[key] = conversation_raw[key]
                if index_name == 'wildchat':
                    conversation['conversation_id'] = conversation_raw['conversation'][0]['turn_identifier']
                conversation_id = conversation['conversation_id']
                if conversation_id not in conversation_ids:
                    conversations.append(conversation)
                    conversation_ids.add(conversation_id)
        if len(conversations) < search_expansion_limit and len(indices_to_search) > 0:
            for index_name in indices_to_search:
                # Execute search query
                search_query, any_filters_ = build_query_for_index(index_name, filters, contains, 0, max(1, search_expansion_limit // len(indices_to_search)))
                response = es.search(index=index_name, body=search_query)
                conversations_raw = [hit['_source'] for hit in response['hits']['hits']]

                for conversation_raw in conversations_raw:
                    conversation = {}
                    conversation['dataset'] = index_name
                    for key in ['conversation', 'conversation_id']:
                        if key in conversation_raw:
                            conversation[key] = conversation_raw[key]
                    if index_name == 'wildchat':
                        conversation['conversation_id'] = conversation_raw['conversation'][0]['turn_identifier']
                    conversation_id = conversation['conversation_id']
                    if conversation_id not in conversation_ids:
                        conversations.append(conversation)
                        conversation_ids.add(conversation_id)
    #conversations = [hit['_source']['conversation'] for hit in response['hits']['hits']]

    conversation_embeddings = {}
    print ('#Matched Conversation:', len(conversations))
    for conversation in conversations:
        dataset = conversation['dataset']
        conversation_id = conversation['conversation_id']
        umap_database_name = f'umap_{visualization_language}_{dataset}_cache.db'
        embed_database_name = f'{dataset}_embeddings_cache.db'
        #create_database(umap_database_name)
        hit, embedding_2d = retrieve(umap_database_name, conversation_id)
        if not hit:
            print ('not hit')
            #import pdb; pdb.set_trace()
            conversation_text = conversation['conversation'][0]['content']
            conversation_text = conversation_text.strip()
            if not conversation_text:
                continue
            embedding = get_embedding_with_cache(embed_database_name, conversation_id, conversation_text, model='text-embedding-3-small')
            embedding_2d = umap_encoder(np.array([embedding])).numpy()[0]
            insert_or_update(umap_database_name, conversation_id, '', [float(embedding_2d[0]), float(embedding_2d[1])])
        conversation_embeddings[str(conversation_id)] = {'i': conversation_id, 'e': [round(float(embedding_2d[0]), 4), round(float(embedding_2d[1]), 4)], 'c': conversation['conversation'][0]['content'], 'd': dataset}
    return jsonify(conversation_embeddings)


@app.route("/embeddings/<language>")
@app.route("/embeddings")
def embeddings(language=None):
    data = _data()

    contains = request.args.get('contains', '')
    # Construct the Elasticsearch query
    filters = {
        "toxic": request.args.get('toxic', ''),
        "redacted": request.args.get('redacted', ''),
        "model": request.args.get('model', ''),
        "hashed_ip": request.args.get('hashed_ip', ''),
        "language": request.args.get('language', ''),
        "country": request.args.get('country', ''),
        "state": request.args.get('state', ''),
        "min_turns": request.args.get('min_turns', ''),
        "search_expansion_limit": request.args.get('search_expansion_limit', ''),
        "conversation_id": request.args.get('conversation_id', '')
    }
    #if language:
    #    filters['language'] = language.capitalize()
    #    any_filters = True
    any_filters = False
    for key in filters:
        if filters[key]:
            any_filters = True
    if contains:
        any_filters = True
    #must_clauses = []
    #if must_clauses:
    #    any_filters = True
    #else:
    #    any_filters = False
    #data["papers"] = site_data["papers"]
    data.update({
        "contains": contains,
        "filters": filters,
        "any_filters": any_filters,
        "visualization_language": language or "all"
    })
    return render_template("embeddings.html", **data)

def extract_list_field(v, key):
    value = v.get(key, "")
    if isinstance(value, list):
        return value
    else:
        return value.split("|")

@app.route("/conversation/wildchat/<int:turn_identifier>")
def conversation_wildchat(turn_identifier):
    data = _data()
    search_query = {
        "query": {
            "nested": {
                "path": "conversation",
                "query": {
                    "term": {
                        "conversation.turn_identifier": turn_identifier
                    }
                }
            }
        }
    }

    response = es.search(index="wildchat", body=search_query)
    if not response['hits']['hits']:
        return render_template("error.html", message="Conversation not found."), 404

    # Extract the conversation and check the first turn identifier
    conversation = response['hits']['hits'][0]['_source']
    first_turn_identifier = conversation['conversation'][0]['turn_identifier']

    # If the turn_identifier is not the first turn, redirect to the first turn's page
    if turn_identifier != first_turn_identifier:
        return redirect(url_for('conversation_wildchat', turn_identifier=first_turn_identifier))

    conversation['conversation_id'] = first_turn_identifier
    conversation['dataset'] = 'wildchat'
    data["conversation"] = conversation
    data["from_page"] = request.args.get('from', 'filter')
    data["visualization_language"] = request.args.get('lang', 'all')
    return render_template("conversation.html", **data)

@app.route("/conversation/lmsyschat/<string:conversation_id>")
def conversation_lmsyschat(conversation_id):
    data = _data()
    search_query = {
        "query": {
            "term": {"conversation_id": conversation_id}
        }
    }

    response = es.search(index="lmsyschat", body=search_query)
    if not response['hits']['hits']:
        return render_template("error.html", message="Conversation not found."), 404

    # Extract the conversation and check the first turn identifier
    conversation = response['hits']['hits'][0]['_source']
    conversation['dataset'] = 'lmsyschat'
    conversation['toxic'] = any([item['flagged'] for item in conversation['openai_moderation']])
    data["conversation"] = conversation
    data["from_page"] = request.args.get('from', 'filter')
    data["visualization_language"] = request.args.get('lang', 'all')
    return render_template("conversation.html", **data)



if __name__ == "__main__":
    debug_val = False
    if os.getenv("FLASK_DEBUG") == "True":
        debug_val = True

    app.run(port=8080, debug=debug_val, extra_files=extra_files, host='0.0.0.0')
