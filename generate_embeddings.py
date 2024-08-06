import os
import hashlib
import json
import sqlite3
import tqdm
import tiktoken
from datasets import load_dataset
from sklearn.manifold import TSNE
import numpy as np

from openai import OpenAI
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Database setup for caching
def create_database():
    conn = sqlite3.connect('embeddings_cache.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS embeddings_cache
                 (key TEXT PRIMARY KEY, prompt TEXT, embedding TEXT)''')
    conn.commit()
    conn.close()

create_database()

def insert_or_update(key, prompt, embedding):
    conn = sqlite3.connect('embeddings_cache.db')
    c = conn.cursor()
    c.execute('''INSERT OR REPLACE INTO embeddings_cache
                 (key, prompt, embedding) VALUES (?, ?, ?)''', 
                 (key, prompt, json.dumps(embedding)))
    conn.commit()
    conn.close()

def retrieve(key):
    conn = sqlite3.connect('embeddings_cache.db')
    c = conn.cursor()
    c.execute("SELECT prompt, embedding FROM embeddings_cache WHERE key=?", (key,))
    result = c.fetchone()
    conn.close()
    if result:
        return True, json.loads(result[1])
    else:
        return False, None

tokenizer = tiktoken.get_encoding('cl100k_base')
def get_embedding_with_cache(prompt, model='text-embedding-3-small'):

    key = hashlib.sha256(json.dumps({'prompt': prompt, 'model': model}).encode('utf-8')).hexdigest()
    hit, embedding = retrieve(key)
    if not hit:
        # Tokenize and truncate if necessary
        tokens = tokenizer.encode(prompt)
        if len(tokens) > 8192:
            #import pdb; pdb.set_trace()
            tokens = tokens[:8192]
            prompt = tokenizer.decode(tokens)
        embedding = client.embeddings.create(input = [prompt], model=model).data[0].embedding
        insert_or_update(key, json.dumps(prompt), embedding)
    else:
        print('Cache hit')
    return embedding

# Load dataset
dataset = load_dataset('allenai/WildChat-1M', split='train')

# Process the first 1000 conversations
embeddings = []
conversation_ids = []
i = -1
for conversation in tqdm.tqdm(dataset):
    i += 1
    if i >= 1000:
        break
    
    conversation_text = ''
    for turn in conversation['conversation']:
        conversation_text += f"[{turn['role'].upper()}]: {turn['content']}\n"
    conversation_text = conversation_text.strip()
    #import pdb; pdb.set_trace()
    
    embedding = get_embedding_with_cache(conversation_text, model='text-embedding-3-small')
    embeddings.append(embedding)
    conversation_ids.append(conversation['conversation'][0]['turn_identifier'])

# Use TSNE to reduce dimensions
embeddings = np.array(embeddings)
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# Save the reduced embeddings
reduced_embeddings = [{'id': cid, 'pos': [float(x), float(y)]} for cid, (x, y) in zip(conversation_ids, embeddings_2d)]

with open('reduced_embeddings.json', 'w') as f:
    json.dump(reduced_embeddings, f)

print("Reduced embeddings saved to 'reduced_embeddings.json'")
