import json
from umap.parametric_umap import load_ParametricUMAP
import os
import sqlite3
import numpy as np
from umap.parametric_umap import ParametricUMAP
from tqdm import tqdm
import gzip
import random

def gzip_file(input_file, output_file):
    with open(input_file, 'rb') as f_in:
        with gzip.open(output_file, 'wb') as f_out:
            f_out.writelines(f_in)

def retrieve_embedding(db_name, key):
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    c.execute("SELECT embedding FROM cache WHERE key=?", (key,))
    result = c.fetchone()
    conn.close()
    if result:
        return json.loads(result[0])
    else:
        return None

def process_english_data():
    # Load existing JSON data
    with open('static/english/wildchat_embeddings_all.json', 'r') as f:
        wildchat_data = json.load(f)
    with open('static/english/lmsyschat_embeddings_all.json', 'r') as f:
        lmsyschat_data = json.load(f)

    print(f"Loaded {len(wildchat_data)} WildChat and {len(lmsyschat_data)} LMSYS-Chat samples")

    # Retrieve embeddings
    wildchat_embed_db = 'wildchat_embeddings_cache.db'
    lmsyschat_embed_db = 'lmsyschat_embeddings_cache.db'

    embeddings = []
    valid_samples = {'wildchat': [], 'lmsyschat': []}

    for item in tqdm(wildchat_data, desc="Retrieving WildChat embeddings"):
        embedding = retrieve_embedding(wildchat_embed_db, item['i'])
        if embedding:
            embeddings.append(embedding)
            valid_samples['wildchat'].append(item)

    for item in tqdm(lmsyschat_data, desc="Retrieving LMSYS-Chat embeddings"):
        embedding = retrieve_embedding(lmsyschat_embed_db, item['i'])
        if embedding:
            embeddings.append(embedding)
            valid_samples['lmsyschat'].append(item)

    print(f"Valid samples: WildChat: {len(valid_samples['wildchat'])}, LMSYS-Chat: {len(valid_samples['lmsyschat'])}")

    # Run UMAP with n_neighbors=15
    #umap = ParametricUMAP(n_components=2, n_neighbors=30, min_dist=0.3, metric='cosine')
    #umap = ParametricUMAP(n_components=2, n_neighbors=50, metric='cosine')
    #umap = ParametricUMAP(n_components=2, n_neighbors=30, min_dist=0.6, metric='cosine')
    #umap = ParametricUMAP(n_components=2, n_neighbors=30, metric='cosine')
    #umap = ParametricUMAP(n_components=2, n_neighbors=30, min_dist=0.9, metric='cosine')
    umap = ParametricUMAP(n_components=2, n_neighbors=10, metric='cosine')
    #umap.n_training_epochs = 0.5
    umap_embeddings = umap.fit_transform(embeddings)

    # Save UMAP model
    os.makedirs('umap_model/debugn10', exist_ok=True)
    if hasattr(umap, "_raw_data"):
        del umap._raw_data
    if hasattr(umap, "knn_search_index") and hasattr(umap.knn_search_index, "_raw_data"):
        del umap.knn_search_index._raw_data
    umap.save('umap_model/debugn10')

    # Create new UMAP cache databases
    #for db_path in ['umap_debugmin0_wildchat_cache.db', 'umap_debugmin0_lmsyschat_cache.db']:
    #    conn = sqlite3.connect(db_path)
    #    c = conn.cursor()
    #    c.execute('''CREATE TABLE IF NOT EXISTS cache
    #                 (key TEXT PRIMARY KEY, prompt TEXT, embedding TEXT)''')
    #    conn.commit()
    #    conn.close()

    # Save new UMAP embeddings and create JSON files
    os.makedirs('static/debugn10', exist_ok=True)

    # Create sets of subsampled IDs
    with open('static/english/wildchat_embeddings.json', 'r') as f:
        wildchat_data_subsampled = json.load(f)
    with open('static/english/lmsyschat_embeddings.json', 'r') as f:
        lmsyschat_data_subsampled = json.load(f)
    wildchat_subsampled_ids = set(item['i'] for item in wildchat_data_subsampled)
    lmsyschat_subsampled_ids = set(item['i'] for item in lmsyschat_data_subsampled)

    for dataset_name in ['wildchat', 'lmsyschat']:
        if dataset_name == 'wildchat':
            subsampled_ids = wildchat_subsampled_ids
        else:
            subsampled_ids = lmsyschat_subsampled_ids
        json_data = []
        json_data_subsampled = []
        start_index = 0 if dataset_name == 'wildchat' else len(valid_samples['wildchat'])
        
        for i, item in enumerate(valid_samples[dataset_name], start=start_index):
            umap_embedding = umap_embeddings[i].tolist()
            
            # Save to UMAP cache
            #db_path = f'umap_debugmin0_{dataset_name}_cache.db'
            #conn = sqlite3.connect(db_path)
            #c = conn.cursor()
            #c.execute('''INSERT OR REPLACE INTO cache
            #             (key, prompt, embedding) VALUES (?, ?, ?)''', 
            #             (item['i'], '', json.dumps(umap_embedding)))
            #conn.commit()
            #conn.close()
            new_item = {
                'i': item['i'],
                'e': [round(float(umap_embedding[0]), 4), round(float(umap_embedding[1]), 4)],
                'c': item['c'],
                'd': dataset_name
            }
            json_data.append(new_item)

            if item['i'] in subsampled_ids:
                json_data_subsampled.append(new_item)
            

        
        # Save full JSON
        with open(f'static/debugn10/{dataset_name}_embeddings_all.json', 'w') as f:
            json.dump(json_data, f)
        
        # Save and gzip subsampled JSON
        #subsampled_json_data = random.sample(json_data, min(1500, len(json_data)))

        subsampled_json_path = f'static/debugn10/{dataset_name}_embeddings.json'
        with open(subsampled_json_path, 'w') as f:
            json.dump(json_data_subsampled, f)
        gzip_file(subsampled_json_path, f'{subsampled_json_path}.gz')

    print("Processing complete!")

if __name__ == "__main__":
    process_english_data()
