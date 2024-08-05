# pylint: disable=global-statement,redefined-outer-name
import argparse
import csv
import glob
import json
import os
from markupsafe import Markup, escape
import sqlite3
from openai import OpenAI
import numpy as np
import joblib
import tiktoken
from sklearn.decomposition import PCA


def nl2br(value):
    escaped_value = escape(value)
    return Markup(escaped_value.replace('\n', Markup('<br>')))

import yaml
from flask import Flask, jsonify, redirect, render_template, send_from_directory, request, url_for
from flask_frozen import Freezer
from flaskext.markdown import Markdown
from elasticsearch import Elasticsearch, helpers

site_data = {}
by_uid = {}

es = Elasticsearch('https://localhost:9200', basic_auth=('elastic', os.getenv('ES_PASSWD')), ssl_assert_fingerprint=os.getenv('ES_FINGERPRINT'))

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
tokenizer = tiktoken.get_encoding('cl100k_base')

# Load the PCA model
pca = joblib.load('pca_model.pkl')


def create_database(db_name):
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS cache
                 (key TEXT PRIMARY KEY, prompt TEXT, embedding TEXT)''')
    conn.commit()
    conn.close()

create_database('embeddings_cache.db')
create_database('pca_cache.db')

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
    c.execute("SELECT prompt, embedding FROM cache WHERE key=?", (key,))
    result = c.fetchone()
    conn.close()
    if result:
        return True, json.loads(result[1])
    else:
        return False, None


def get_embedding_with_cache(conversation_id, prompt, model='text-embedding-3-small'):
    key = conversation_id
    hit, embedding = retrieve('embeddings_cache.db', key)
    if not hit:
        # Tokenize and truncate if necessary
        tokens = tokenizer.encode(prompt)
        if len(tokens) > 8192:
            tokens = tokens[:8192]
            prompt = tokenizer.decode(tokens)
        embedding = client.embeddings.create(input=[prompt], model=model).data[0].embedding
        insert_or_update('embeddings_cache.db', key, json.dumps(prompt), embedding)
    else:
        print('Cache hit for embedding')
    return embedding

def main(site_data_path):
    global site_data, extra_files
    extra_files = ["README.md"]
    # Load all for your sitedata one time.
    for f in glob.glob(site_data_path + "/*"):
        extra_files.append(f)
        name, typ = f.split("/")[-1].split(".")
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
    size = 30
    from_ = (page - 1) * size

    if from_ >= 10000:
        return render_template("error.html", message="You cannot navigate beyond the 10,000th result. Please refine your search by going to earlier pages.")
    if from_+size > 10000:
        size_ = 10000 - from_
    else:
        size_ = size

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
        "conversation_id": request.args.get('conversation_id', '')
    }

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
        must_clauses.append({"term": {"toxic": filters['toxic'] == 'true'}})
    if filters['redacted']:
        must_clauses.append({"term": {"redacted": filters['redacted'] == 'true'}})
    if filters['model']:
        must_clauses.append({"term": {"model": filters['model']}})
    if filters['hashed_ip']:
        must_clauses.append({"term": {"hashed_ip": filters['hashed_ip']}})
    if filters['language']:
        must_clauses.append({"term": {"language": filters['language'].title()}})
    if filters['country']:
        must_clauses.append({"term": {"country": filters['country']}})
    if filters['state']:
        must_clauses.append({"term": {"state": filters['state']}})
    if filters['min_turns']:
        must_clauses.append({"range": {"turn": {"gte": int(filters['min_turns'])}}})
    if filters['conversation_id']:
        must_clauses.append({"nested": {"path": "conversation", "query": {"term": {"conversation.turn_identifier": filters['conversation_id']}}}})

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

    # Execute search query
    response = es.search(index="wildchat", body=search_query)
    conversations = [hit['_source'] for hit in response['hits']['hits']]
    total = response['hits']['total']['value']
    #total_pages = (total // size) + 1
    total_pages = (total + size - 1) // size

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

    #filters = {
    #    "contains": request.args.get('contains', ''),
    #    "toxic": request.args.get('toxic', ''),
    #    "redacted": request.args.get('redacted', ''),
    #    "model": request.args.get('model', ''),
    #    "hashed_ip": request.args.get('hashed_ip', ''),
    #    "language": request.args.get('language', ''),
    #    "country": request.args.get('country', ''),
    #    "state": request.args.get('state', ''),
    #    "min_turns": request.args.get('min_turns', ''),
    #    "conversation_id": request.args.get('conversation_id', '')
    #}
    print (filters)

    must_clauses = []

    if filters['contains']:
        must_clauses.append({"nested": {"path": "conversation", "query": {"match_phrase": {"conversation.content": filters['contains']}}}})
    if filters['toxic']:
        must_clauses.append({"term": {"toxic": filters['toxic'] == 'true'}})
    if filters['hashed_ip']:
        must_clauses.append({"term": {"hashed_ip": filters['hashed_ip']}})
    if filters['language']:
        must_clauses.append({"term": {"language": filters['language'].title()}})
    if filters['country']:
        must_clauses.append({"term": {"country": filters['country']}})
    if filters['state']:
        must_clauses.append({"term": {"state": filters['state']}})
    if filters['min_turns']:
        must_clauses.append({"range": {"turn": {"gte": int(filters['min_turns'])}}})
    if filters['model']:
        must_clauses.append({"term": {"model": filters['model']}})
    if filters['redacted']:
        must_clauses.append({"term": {"redacted": filters['redacted'] == 'true'}})
    if filters['conversation_id']:
        must_clauses.append({"nested": {"path": "conversation", "query": {"term": {"conversation.turn_identifier": filters['conversation_id']}}}})

    print (must_clauses)
    search_query = {"query": {"bool": {"must": must_clauses}}, 'size': 100}

    response = es.search(index="wildchat", body=search_query)
    conversations = [hit['_source']['conversation'] for hit in response['hits']['hits']]

    conversation_embeddings = {}
    print ('fsfs', len(conversations))
    for conversation in conversations:
        conversation_id = conversation[0]['turn_identifier']
        hit, embedding = retrieve('pca_cache.db', conversation_id)
        if not hit:
            conversation_text = ''
            for turn in conversation:
                conversation_text += f"[{turn['role'].upper()}]: {turn['content']}\n"
            conversation_text = conversation_text.strip()
            embedding = get_embedding_with_cache(conversation_id, conversation_text, model='text-embedding-3-small')
            embedding_2d = pca.transform(np.array([embedding]))[0]
            insert_or_update('pca_cache.db', conversation_id, '', [float(embedding_2d[0]), float(embedding_2d[1])])
            conversation_embeddings[conversation_id] = {'pos': [float(embedding_2d[0]), float(embedding_2d[1])]}
        else:
            print ('hit')
            conversation_embeddings[conversation_id] = {'pos': embedding}

    return jsonify(conversation_embeddings)

@app.route("/conversation_vis.html")
def conversation_vis():
    data = _data()
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
        "conversation_id": request.args.get('conversation_id', '')
    }
    must_clauses = []
    if must_clauses:
        any_filters = True
    else:
        any_filters = False
    #data["papers"] = site_data["papers"]
    data.update({
        "contains": contains,
        "filters": filters,
        "any_filters": any_filters
    })
    return render_template("conversation_vis.html", **data)
@app.route("/papers.json")
def paper_json():
    json = []
    for v in site_data["papers"]:
        json.append(format_paper(v))
    return jsonify(json)

@app.route("/serve_<path>.json")
def serve(path):
    return jsonify(site_data[path])

def extract_list_field(v, key):
    value = v.get(key, "")
    if isinstance(value, list):
        return value
    else:
        return value.split("|")

def format_paper(v):
    list_keys = ["authors", "keywords", "sessions"]
    list_fields = {}
    for key in list_keys:
        list_fields[key] = extract_list_field(v, key)

    return {
        "UID": v["UID"],
        "title": v["title"],
        "forum": v["UID"],
        "authors": list_fields["authors"],
        "keywords": list_fields["keywords"],
        "abstract": v["abstract"],
        "TLDR": v["abstract"],
        "recs": [],
        "sessions": list_fields["sessions"],
        # links to external content per poster
        "pdf_url": v.get("pdf_url", ""),  # render poster from this PDF
        "code_link": "https://github.com/Mini-Conf/Mini-Conf",  # link to code
        "link": "https://arxiv.org/abs/2007.12238",  # link to paper
    }

@app.route("/conversation/<int:turn_identifier>")
def conversation(turn_identifier):
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
        return redirect(url_for('conversation', turn_identifier=first_turn_identifier))

    data["conversation"] = conversation
    return render_template("conversation.html", **data)


if __name__ == "__main__":
    debug_val = False
    if os.getenv("FLASK_DEBUG") == "True":
        debug_val = True

    app.run(port=8080, debug=debug_val, extra_files=extra_files, host='0.0.0.0')
