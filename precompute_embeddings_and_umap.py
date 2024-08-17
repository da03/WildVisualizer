import json
import random
import os
import sqlite3
from openai import OpenAI
import numpy as np
import joblib
from umap.parametric_umap import ParametricUMAP
import tiktoken
from sklearn.preprocessing import StandardScaler
from datasets import load_dataset
from tqdm import tqdm

LANGUAGES = ['all', 'english', 'russian', 'chinese', 'spanish', 'german', 'french', 'portuguese', 'italian', 'japanese', 'korean']

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
tokenizer = tiktoken.get_encoding('cl100k_base')

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
    else:
        print('Cache hit for embedding')
    return embedding

def conditional_reservoir_sample(dataset, n, language=None):
    reservoir = []
    count = 0
    for item in tqdm(dataset, desc=f"Sampling {language if language else 'all'}"):
        if language is None or item['language'].lower() == language.lower():
            count += 1
            if len(reservoir) < n:
                reservoir.append(item)
            else:
                j = random.randint(0, count - 1)
                if j < n:
                    reservoir[j] = item
    return reservoir

def process_language(wildchat_dataset, lmsyschat_dataset, language):
    wildchat_embed_db = 'wildchat_embeddings_cache.db'
    lmsyschat_embed_db = 'lmsyschat_embeddings_cache.db'
    
    random.seed(1234)
    wildchat_sampled = conditional_reservoir_sample(wildchat_dataset, 15000, language if language != 'all' else None)
    random.seed(1234)
    lmsyschat_sampled = conditional_reservoir_sample(lmsyschat_dataset, 15000, language if language != 'all' else None)
    random.seed(1234)
    wildchat_sampled = wildchat_sampled[:1500]
    lmsyschat_sampled = lmsyschat_sampled[:1500]
    
    print(f"Sampled {len(wildchat_sampled)} from WildChat and {len(lmsyschat_sampled)} from LMSYS-Chat for {language}")
    
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
    
    scaler = StandardScaler()
    umap = ParametricUMAP(n_components=2)
    scaled_embeddings = scaler.fit_transform(embeddings)
    umap_embeddings = umap.fit_transform(scaled_embeddings)
    
    os.makedirs(f'umap_model/{language}', exist_ok=True)
    joblib.dump(scaler, f'umap_model/{language}/scaler.pkl')
    umap.save(f'umap_model/{language}')
    
    wildchat_umap_db = create_database(f'umap_{language}_wildchat_cache.db')
    lmsyschat_umap_db = create_database(f'umap_{language}_lmsyschat_cache.db')
    
    for dataset_name in ['wildchat', 'lmsyschat']:
        json_data = []
        start_index = 0 if dataset_name == 'wildchat' else len(valid_samples['wildchat'])
        
        for i, item in enumerate(valid_samples[dataset_name], start=start_index):
            conversation_id = item['conversation'][0]['turn_identifier'] if dataset_name == 'wildchat' else item['conversation_id']
            umap_embedding = umap_embeddings[i].tolist()
            
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
        with open(f'static/{language}/{dataset_name}_embeddings.json', 'w') as f:
            json.dump(subsampled_json_data, f)

# Main processing loop
print("Loading datasets...")
wildchat_dataset = load_dataset("allenai/WildChat-1M-Full")['train']
lmsyschat_dataset = load_dataset("lmsys/LMSYS-Chat-1M")['train']

for language in LANGUAGES:
    print(f"Processing {language}...")
    process_language(wildchat_dataset, lmsyschat_dataset, language)

print("Processing complete!")
