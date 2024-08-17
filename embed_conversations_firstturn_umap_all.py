import os
import sys
import tqdm
import hashlib
import json
import random
import sqlite3
import tqdm
import tiktoken
from datasets import load_dataset
from sklearn.decomposition import PCA
import numpy as np
import joblib
from openai import OpenAI
import tiktoken
import tqdm

# Initialize OpenAI API client and tokenizer
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
tokenizer = tiktoken.get_encoding('cl100k_base')

# Function to create SQLite database for caching embeddings
def create_database(db_name):
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS cache
                 (key TEXT PRIMARY KEY, prompt TEXT, embedding TEXT)''')
    conn.commit()
    conn.close()

# Load datasets
datasets = {
    'wildchat': load_dataset('allenai/WildChat-1M-Full'),
    'lmsyschat': load_dataset('lmsys/LMSYS-Chat-1M')
}

# Initialize prefix sets and database names
prefix_sets = {name: set() for name in datasets}
embedding_db_names = {name: f'{name}_embeddings_cache.db' for name in datasets}
umap_db_names = {name: f'{name}_umap_cache.db' for name in datasets}
json_files = {name: f'{name}_umap_embeddings.json' for name in datasets}
# Create separate SQLite databases for each dataset
for db_name in embedding_db_names.values():
    create_database(db_name)
for db_name in umap_db_names.values():
    create_database(db_name)
# Function to insert or update cached embeddings
def insert_or_update(db_name, key, prompt, embedding):
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    c.execute('''INSERT OR REPLACE INTO cache
                 (key, prompt, embedding) VALUES (?, ?, ?)''', 
                 (key, prompt, json.dumps(embedding)))
    conn.commit()
    conn.close()

# Function to retrieve cached embeddings
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

# Function to get embedding with caching
def get_embedding_with_cache(db_name, conversation_id, prompt, model='text-embedding-3-small'):
    key = conversation_id
    hit, embedding = retrieve(db_name, key)
    if not hit:
        # Tokenize and truncate if necessary
        tokens = tokenizer.encode(prompt)
        if len(tokens) > 8192:
            tokens = tokens[:8192]
            prompt = tokenizer.decode(tokens)
        embedding = client.embeddings.create(input=[prompt], model=model).data[0].embedding
        insert_or_update(db_name, key, json.dumps(prompt), embedding)
    return embedding

# Function for reservoir sampling
def reservoir_sampling(iterable, k):
    random.seed(1234)
    sample = []
    for i, item in enumerate(tqdm.tqdm(iterable)):
        if i < k:
            sample.append(item)
        else:
            #return sample
            j = random.randint(0, i)
            if j < k:
                sample[j] = item
    return sample



# Step 1: Collect embeddings from both datasets
all_embeddings = []
embedding_data_by_dataset = {}

for dataset_name, dataset in datasets.items():
    selected_conversations = reservoir_sampling(dataset['train'], 10000)
    embedding_data_by_dataset[dataset_name] = []

    for conversation in tqdm.tqdm(selected_conversations, desc=f"Processing {dataset_name}"):
        # Assume the first turn with 'role' == 'user' is the first user turn
        first_user_turn = conversation['conversation'][0]['content']
        if not first_user_turn:
            continue
        user_turn_content = first_user_turn
        # Check for uniqueness of the content
        if user_turn_content in prefix_sets[dataset_name]:
            continue
        prefix_sets[dataset_name].add(user_turn_content)

        # Get the embedding for the first user turn
        if dataset_name == 'wildchat':
            conversation_id = conversation['conversation'][0]['turn_identifier']
        else:
            conversation_id = conversation['conversation_id']
        embedding = get_embedding_with_cache(embedding_db_names[dataset_name], conversation_id, user_turn_content)

        # Store the embedding for PCA fitting later
        all_embeddings.append(embedding)
        embedding_data_by_dataset[dataset_name].append({
            'i': conversation_id,
            'e': embedding,
            'c': user_turn_content
        })
print ('step 1 done')
sys.stdout.flush()
# Step 2: Fit PCA on all collected embeddings
all_embeddings = np.array(all_embeddings)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

#umap = PCA(n_components=2)



#from sklearn.manifold import TSNE
#tsne = TSNE(n_components=2, random_state=42)
#embeddings_2ds = tsne.fit_transform(all_embeddings)
#import umap
from umap.parametric_umap import ParametricUMAP

scaler = StandardScaler()
umap = ParametricUMAP(n_components=2)

all_embeddings = scaler.fit_transform(all_embeddings)
#embeddings_2ds = umap.fit_transform(all_embeddings)
umap.fit(all_embeddings)



# Save the PCA model
#joblib.dump(umap, 'umap_model_both.pkl')
save_dir = 'umap_model'
os.makedirs(save_dir, exist_ok=True)
joblib.dump(scaler, os.path.join(save_dir, 'scaler.pkl'))
umap.save(save_dir)
# Step 3: Project embeddings, save to JSON files, and cache in SQLite databases
i = 0
for dataset_name, embedding_data in embedding_data_by_dataset.items():
    for data in tqdm.tqdm(embedding_data):
        embedding_2d = umap.transform(scaler.transform([data['e']]))[0]
        #embedding_2d = embeddings_2ds[i]
        i += 1
        data['e'] = [float(embedding_2d[0]), float(embedding_2d[1])]

        # Save the 2D embedding into the SQLite database
        insert_or_update(umap_db_names[dataset_name], data['i'], '', data['e'])
        data['e'] = [round(float(embedding_2d[0]), 4), round(float(embedding_2d[1]), 4)]
        
    # Save the embedding data to JSON
    with open(json_files[dataset_name], 'w') as f:
        json.dump(embedding_data, f)
