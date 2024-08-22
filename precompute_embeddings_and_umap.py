import json
import copy
import random
import os
import sqlite3
from collections import defaultdict
from openai import OpenAI
import numpy as np
import joblib
from umap.parametric_umap import ParametricUMAP
import tiktoken
from sklearn.preprocessing import StandardScaler
from datasets import load_dataset
from tqdm import tqdm
import gzip
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

LANGUAGES = ['all', 'english', 'chinese', 'russian', 'spanish', 'french', 'portuguese', 'german', 'italian', 'turkish', 'arabic', 'japanese', 'korean', 'polish', 'vietnamese']

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
tokenizer = tiktoken.get_encoding('cl100k_base')


def gzip_file(input_file, output_file):
    with open(input_file, 'rb') as f_in:
        with gzip.open(output_file, 'wb') as f_out:
            f_out.writelines(f_in)

def create_database(db_name):
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS cache
                 (key TEXT PRIMARY KEY, prompt TEXT, embedding TEXT)''')
    conn.commit()
    conn.close()
    return db_name

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

#def conditional_reservoir_sample(dataset, n, language=None):
#    reservoir = []
#    count = 0
#    for item in tqdm(dataset, desc=f"Sampling {language if language else 'all'}"):
#        if language is None or item['language'].lower() == language.lower():
#            count += 1
#            if len(reservoir) < n:
#                reservoir.append(item)
#            else:
#                j = random.randint(0, count - 1)
#                if j < n:
#                    reservoir[j] = item
#    return reservoir



def process_item(item, dataset_name, embed_db):
    first_turn = item['conversation'][0]['content'].strip()
    if not first_turn:
        return None
    if dataset_name == 'wildchat':
        conversation_id = item['conversation'][0]['turn_identifier']
    else:
        conversation_id = item['conversation_id']
    embedding = get_embedding_with_cache(embed_db, conversation_id, first_turn)

def process_language(wildchat_sampled, lmsyschat_sampled, language):
    wildchat_embed_db = 'wildchat_embeddings_cache.db'
    lmsyschat_embed_db = 'lmsyschat_embeddings_cache.db'
    
    #random.seed(1234)
    #wildchat_sampled = conditional_reservoir_sample(wildchat_dataset, 10000, language if language != 'all' else None)
    #random.seed(1234)
    #lmsyschat_sampled = conditional_reservoir_sample(lmsyschat_dataset, 10000, language if language != 'all' else None)
    random.seed(1234)
    #wildchat_sampled = wildchat_sampled[:1500]
    #lmsyschat_sampled = lmsyschat_sampled[:1500]
    all_items = [(item, 'wildchat', wildchat_embed_db) for item in wildchat_sampled] + \
                [(item, 'lmsyschat', lmsyschat_embed_db) for item in lmsyschat_sampled]
    
    print(f"Sampled {len(wildchat_sampled)} from WildChat and {len(lmsyschat_sampled)} from LMSYS-Chat for {language}")

    # Use ThreadPoolExecutor to process items in parallel
    with ThreadPoolExecutor(max_workers=5) as executor:  # Adjust max_workers as needed
        future_to_item = {executor.submit(process_item, item, dataset, db): (item, dataset) 
                          for item, dataset, db in all_items}
        for future in tqdm(as_completed(future_to_item), total=len(all_items), desc="Pre-Computing embeddings"):
            result = future.result() 
    
    embeddings = []
    valid_samples = {'wildchat': [], 'lmsyschat': []}
    
    for item in tqdm(wildchat_sampled, desc="Computing WildChat embeddings"):
        conversation_id = item['conversation'][0]['turn_identifier']
        first_turn = item['conversation'][0]['content'].strip()
        if not first_turn:
            continue
        embedding = get_embedding_with_cache(wildchat_embed_db, conversation_id, first_turn)
        embeddings.append(embedding)
        valid_samples['wildchat'].append(item)
    
    for item in tqdm(lmsyschat_sampled, desc="Computing LMSYS-Chat embeddings"):
        conversation_id = item['conversation_id']
        first_turn = item['conversation'][0]['content'].strip()
        if not first_turn:
            continue
        embedding = get_embedding_with_cache(lmsyschat_embed_db, conversation_id, first_turn)
        embeddings.append(embedding)
        valid_samples['lmsyschat'].append(item)
    
    print(f"Valid samples after filtering: WildChat: {len(valid_samples['wildchat'])}, LMSYS-Chat: {len(valid_samples['lmsyschat'])}")
    #random.seed(1234)
    #random.shuffle(embeddings)
    #import pdb; pdb.set_trace()
    #scaler = StandardScaler()
    umap = ParametricUMAP(n_components=2, n_neighbors=50, spread=0.5, min_dist=0, metric='cosine', verbose=True)
    #scaled_embeddings = scaler.fit_transform(embeddings)
    #umap_embeddings = umap.fit_transform(scaled_embeddings)
    embeddings_shuffled = copy.deepcopy(embeddings)
    random.seed(1234)
    random.shuffle(embeddings_shuffled)
    umap.fit(embeddings_shuffled)
    
    os.makedirs(f'umap_model/{language}', exist_ok=True)
    #joblib.dump(scaler, f'umap_model/{language}/scaler.pkl')
    if hasattr(umap, "_raw_data"):
        del umap._raw_data
    if hasattr(umap, "knn_search_index") and hasattr(umap.knn_search_index, "_raw_data"):
        del umap.knn_search_index._raw_data
    umap.save(f'umap_model/{language}')

    for model_path in [f'umap_model/{language}/model.pkl', f'umap_model/{language}/parametric_model.keras', f'umap_model/{language}/scaler.pkl']:
        if os.path.exists(model_path):
            print (f'Removing {model_path}')
            os.remove(model_path)

    for db_path in [f'umap_{language}_wildchat_cache.db', f'umap_{language}_lmsyschat_cache.db']:
        if os.path.exists(db_path):
            print (f'Removing {db_path}')
            os.remove(db_path)
  
    wildchat_umap_db = create_database(f'umap_{language}_wildchat_cache.db')
    lmsyschat_umap_db = create_database(f'umap_{language}_lmsyschat_cache.db')
    
    for dataset_name in ['wildchat', 'lmsyschat']:
        json_data = []
        start_index = 0 if dataset_name == 'wildchat' else len(valid_samples['wildchat'])
        
        for i, item in enumerate(valid_samples[dataset_name], start=start_index):
            conversation_id = item['conversation'][0]['turn_identifier'] if dataset_name == 'wildchat' else item['conversation_id']
            umap_embedding = umap.encoder(np.array([embeddings[i]])).numpy()[0].tolist() #umap_embeddings[i].tolist()
            
            if dataset_name == 'wildchat':
                insert_or_update(wildchat_umap_db, conversation_id, '', umap_embedding)
            else:
                insert_or_update(lmsyschat_umap_db, conversation_id, '', umap_embedding)
            
            json_data.append({
                'i': conversation_id,
                'e': [round(float(umap_embedding[0]), 4), round(float(umap_embedding[1]), 4)],
                'c': item['conversation'][0]['content'],
                'd': dataset_name
            })
        
        os.makedirs(f'static/{language}', exist_ok=True)
        with open(f'static/{language}/{dataset_name}_embeddings_all.json', 'w') as f:
            json.dump(json_data, f)
        
        subsampled_json_data = random.sample(json_data, min(1500, len(json_data)))
        subsampled_json_path = f'static/{language}/{dataset_name}_embeddings.json'
        with open(subsampled_json_path, 'w') as f:
            json.dump(subsampled_json_data, f)
        gzip_file(subsampled_json_path, f'{subsampled_json_path}.gz')

## Main processing loop
#print("Loading datasets...")
#wildchat_dataset = load_dataset("allenai/WildChat-1M-Full")['train']
#lmsyschat_dataset = load_dataset("lmsys/LMSYS-Chat-1M")['train']

def multi_language_reservoir_sample(dataset, n_per_language, languages):
    reservoirs = {lang: [] for lang in languages}
    counts = {lang: 0 for lang in languages}
    all_samples = []
    all_count = 0
    
    for item in tqdm(dataset, desc="Sampling across languages"):
        lang = item['language'].lower()
        
        # Sample for 'all' category
        all_count += 1
        if len(all_samples) < n_per_language:
            all_samples.append(item)
        else:
            j = random.randint(0, all_count - 1)
            if j < n_per_language:
                all_samples[j] = item
        
        # Sample for specific language
        if lang in reservoirs:
            counts[lang] += 1
            if len(reservoirs[lang]) < n_per_language:
                reservoirs[lang].append(item)
            else:
                j = random.randint(0, counts[lang] - 1)
                if j < n_per_language:
                    reservoirs[lang][j] = item
    
    reservoirs['all'] = all_samples
    return reservoirs

# In your main processing loop:
print("Loading datasets...")
wildchat_dataset = load_dataset("allenai/WildChat-1M-Full")['train']
lmsyschat_dataset = load_dataset("lmsys/LMSYS-Chat-1M")['train']

# Perform reservoir sampling for all languages in one pass
n_per_language = 10000  # Adjust as needed
wildchat_samples = multi_language_reservoir_sample(wildchat_dataset, n_per_language, LANGUAGES)
lmsyschat_samples = multi_language_reservoir_sample(lmsyschat_dataset, n_per_language, LANGUAGES)

for language in LANGUAGES:
    print(f"Processing {language}...")
    wildchat_sampled = wildchat_samples[language]
    lmsyschat_sampled = lmsyschat_samples[language]
    
    process_language(wildchat_sampled, lmsyschat_sampled, language)

print("Processing complete!")
