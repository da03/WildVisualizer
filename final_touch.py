import os
import json
import shutil
import numpy as np
from tensorflow import keras
from tqdm import tqdm
import glob
import sqlite3
import random
from umap.parametric_umap import ParametricUMAP
import gzip

def gzip_file(input_file, output_file):
    with open(input_file, 'rb') as f_in:
        with gzip.open(output_file, 'wb') as f_out:
            f_out.writelines(f_in)

def insert_or_update(db_name, key, prompt, embedding):
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    c.execute('''INSERT OR REPLACE INTO cache
                 (key, prompt, embedding) VALUES (?, ?, ?)''', 
                 (key, prompt, json.dumps(embedding)))
    conn.commit()
    conn.close()

def create_database(db_name):
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS cache
                 (key TEXT PRIMARY KEY, prompt TEXT, embedding TEXT)''')
    conn.commit()
    conn.close()
    return db_name

def create_or_connect_db(db_name):
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS cache
                 (key TEXT PRIMARY KEY, prompt TEXT, embedding TEXT)''')
    conn.commit()
    conn.close()

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

def update_umap_cache(db_name, key, new_embedding):
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    c.execute('''INSERT OR REPLACE INTO cache
                 (key, prompt, embedding) VALUES (?, ?, ?)''', 
                 (key, '', json.dumps(new_embedding.tolist())))
    conn.commit()
    conn.close()

def remove_debug_folders():
    debug_folders = glob.glob('static/debug*')
    for folder in debug_folders:
        print(f"Removing {folder}")
        shutil.rmtree(folder)
        
        # Remove corresponding database files
        db_files = glob.glob(f'umap_{os.path.basename(folder)}_*_cache.db')
        for db_file in db_files:
            os.remove(db_file)
        
        # Remove corresponding UMAP model
        umap_model_path = f'umap_model/{os.path.basename(folder)}'
        if os.path.exists(umap_model_path):
            shutil.rmtree(umap_model_path)

def process_non_debug_folders():
    folders = [f for f in glob.glob('static/*') if os.path.isdir(f) and not f.startswith('static/debug')]
    
    #for folder in folders:
    #    print(f"Processing {folder}")
    #    language = os.path.basename(folder)
    LANGUAGES = ['english', 'all', 'chinese', 'russian', 'spanish', 'german', 'french', 'portuguese', 'italian', 'japanese', 'korean', 'turkish', 'arabic', 'polish', 'vietnamese']
    LANGUAGES = ['english', 'all']
    LANGUAGES = ['all']
    LANGUAGES = ['all', 'english', 'chinese', 'russian', 'spanish', 'french', 'portuguese', 'german', 'italian', 'turkish', 'arabic', 'japanese', 'korean', 'polish', 'vietnamese']
    for language in LANGUAGES:
        folder = f'static/{language}'
        print(f"Processing {folder}")
        
        embeddings = []
        all_data = {'wildchat': [], 'lmsyschat': []}
        sampled_ids = {'wildchat': set(), 'lmsyschat': set()}
        
        for dataset_name in ['wildchat', 'lmsyschat']:
            json_path_all = f'{folder}/{dataset_name}_embeddings_all.json'
            json_path_sampled = f'{folder}/{dataset_name}_embeddings.json'
            embed_db = f'{dataset_name}_embeddings_cache.db'
            
            # Load all data
            with open(json_path_all, 'r') as f:
                data = json.load(f)

            #random.seed(1234)
            #random.shuffle(data)
            #data = data[:5000]
            
            # Load sampled data to get IDs
            with open(json_path_sampled, 'r') as f:
                sampled_data = json.load(f)
                sampled_ids[dataset_name] = set(item['i'] for item in sampled_data)
            
            for item in tqdm(data, desc=f"Loading {dataset_name} embeddings"):
                embedding = retrieve_embedding(embed_db, item['i'])
                if embedding:
                    embeddings.append(embedding)
                    all_data[dataset_name].append(item)
        
        ## Train new UMAP model
        #random.seed(1234)
        #random.shuffle(embeddings)
        #embeddings = embeddings[:10000]
        ##umap_model = ParametricUMAP(n_components=2, n_neighbors=30, spread=1.0, metric='cosine')
        ##umap_model = ParametricUMAP(n_components=2, n_neighbors=30, spread=0.5, metric='cosine')
        ##umap_model = ParametricUMAP(n_components=2, n_neighbors=50, spread=0.5, min_dist=0, metric='cosine', verbose=True)
        #umap_model = ParametricUMAP(n_components=2, n_neighbors=50, spread=0.3, min_dist=0, metric='cosine', verbose=True)
        #umap_model.fit(embeddings)
        #
        ## Save new UMAP model
        #os.makedirs(f'umap_model/{language}', exist_ok=True)
        #umap_model.save(f'umap_model/{language}')
        
        # Load the saved UMAP encoder
        umap_encoder = keras.models.load_model(os.path.join(f'umap_model/{language}', "encoder.keras"))
        for db_path in [f'umap_{language}_wildchat_cache.db', f'umap_{language}_lmsyschat_cache.db']:
            if os.path.exists(db_path):
                print (f'Removing {db_path}')
                os.remove(db_path)
  
        wildchat_umap_db = create_database(f'umap_{language}_wildchat_cache.db')
        lmsyschat_umap_db = create_database(f'umap_{language}_lmsyschat_cache.db')
        
        # Process and save new data
        for dataset_name in ['wildchat', 'lmsyschat']:
            json_path_all = f'{folder}/{dataset_name}_embeddings_all.json'
            json_path_sampled = f'{folder}/{dataset_name}_embeddings.json'
            umap_cache_db = f'umap_{language}_{dataset_name}_cache.db'
            
            #create_or_connect_db(umap_cache_db)
            
            new_data_all = []
            new_data_sampled = []
            
            for item in tqdm(all_data[dataset_name], desc=f"Processing {dataset_name}"):
                embedding = retrieve_embedding(f'{dataset_name}_embeddings_cache.db', item['i'])
                umap_embedding = umap_encoder(np.array([embedding])).numpy()[0] #.tolist()
                #insert_or_update(umap_cache_db, item['i'], '', umap_embedding)
                #new_item = item.copy()
                #new_item['e'] = [round(float(umap_embedding[0]), 4), round(float(umap_embedding[1]), 4)]
                #new_data_all.append(new_item)
                
                # Update UMAP cache database
                update_umap_cache(umap_cache_db, item['i'], umap_embedding)
                
                ## If this item was in the original sample, add it to the new sample
                #if item['i'] in sampled_ids[dataset_name]:
                #    new_data_sampled.append(new_item)
            
            ## Write new JSON files
            #with open(json_path_all, 'w') as f:
            #    json.dump(new_data_all, f)
            #
            #with open(json_path_sampled, 'w') as f:
            #    json.dump(new_data_sampled, f)
            #gzip_file(json_path_sampled, f'{json_path_sampled}.gz')
            
            print(f"Updated {json_path_all}, {json_path_sampled}, and {umap_cache_db}")
        
        print(f"Finished processing {folder}")

if __name__ == "__main__":
    #remove_debug_folders()
    process_non_debug_folders()
