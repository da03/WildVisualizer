import os
import hashlib
import json
import sqlite3
import tqdm
import tiktoken
from datasets import load_dataset
from sklearn.decomposition import PCA
import numpy as np
import joblib

from openai import OpenAI
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Database setup for caching
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

tokenizer = tiktoken.get_encoding('cl100k_base')
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

# Load dataset
dataset = load_dataset('allenai/WildChat-1M', split='train')

# Process the first 1000 conversations
embeddings = []
conversation_ids = []
i = -1
for conversation in tqdm.tqdm(dataset):
    i += 1
    if i >= 10000:
        break
    
    conversation_id = conversation['conversation'][0]['turn_identifier']
    conversation_text = ''
    for turn in conversation['conversation']:
        conversation_text += f"[{turn['role'].upper()}]: {turn['content']}\n"
    conversation_text = conversation_text.strip()
    
    embedding = get_embedding_with_cache(conversation_id, conversation_text, model='text-embedding-3-small')
    embeddings.append(embedding)
    conversation_ids.append(conversation_id)

# Use PCA to reduce dimensions
embeddings = np.array(embeddings)
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

# Save the PCA model
joblib.dump(pca, 'pca_model.pkl')

# Save the reduced embeddings and cache them
reduced_embeddings = []
for cid, (x, y) in zip(conversation_ids, embeddings_2d):
    reduced_embeddings.append({'id': cid, 'pos': [float(x), float(y)]})
    insert_or_update('pca_cache.db', cid, '', [float(x), float(y)])

with open('reduced_embeddings.json', 'w') as f:
    json.dump(reduced_embeddings, f)

print("Reduced embeddings saved to 'reduced_embeddings.json'")
