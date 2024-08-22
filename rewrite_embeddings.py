import os
import json
import shutil
import numpy as np
from tensorflow import keras
from tqdm import tqdm
import glob
import sqlite3

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

def process_debug_folders():
    debug_folders = glob.glob('static/debugs*')
    
    for folder in debug_folders:
        if folder != 'static/debugs05':
            continue
        print(f"Processing {folder}")
        
        # Load UMAP model
        umap_path = f'umap_model/{os.path.basename(folder)}'
        if not os.path.exists(os.path.join(umap_path, "encoder.keras")):
            print(f"No UMAP model found for {folder}. Skipping.")
            continue
        
        umap_encoder = keras.models.load_model(os.path.join(umap_path, "encoder.keras"))
        
        for dataset_name in ['wildchat', 'lmsyschat']:
            json_path = f'{folder}/{dataset_name}_embeddings.json'
            backup_path = f'{json_path}.backup'
            
            # Backup original file
            shutil.copy2(json_path, backup_path)
            
            # Load JSON data
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Retrieve embeddings and recompute
            embed_db = f'{dataset_name}_embeddings_cache.db'
            umap_cache_db = f'umap_{os.path.basename(folder)}_{dataset_name}_cache.db'
            
            # Backup UMAP cache database
            shutil.copy2(umap_cache_db, f'{umap_cache_db}.backup')
            
            new_data = []
            
            for item in tqdm(data, desc=f"Processing {dataset_name}"):
                embedding = retrieve_embedding(embed_db, item['i'])
                if embedding:
                    #new_embedding = umap_encoder.predict(np.array([embedding]), batch_size=1)[0]
                    new_embedding = umap_encoder(np.array([embedding])).numpy()[0]
                    new_item = item.copy()
                    new_item['e'] = [round(float(new_embedding[0]), 4), round(float(new_embedding[1]), 4)]
                    new_data.append(new_item)
                    
                    # Update UMAP cache database
                    create_or_connect_db(umap_cache_db)
                    update_umap_cache(umap_cache_db, item['i'], new_embedding)
                else:
                    print(f"Warning: No embedding found for {item['i']}")
            
            # Write new JSON file
            with open(json_path, 'w') as f:
                json.dump(new_data, f)
            
            print(f"Updated {json_path} and {umap_cache_db}")
        
        print(f"Finished processing {folder}")

if __name__ == "__main__":
    process_debug_folders()
