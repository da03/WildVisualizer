import openai
import sqlite3
import json
import os
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
from sklearn.manifold import TSNE
import tiktoken
import joblib

# Set your OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Connect to SQLite database for embeddings
db_file = 'embedding_cache.db'
conn = sqlite3.connect(db_file)
c = conn.cursor()

# Connect to SQLite database for t-SNE results
tsne_db_file = 'tsne_cache.db'
tsne_conn = sqlite3.connect(tsne_db_file)
tsne_c = tsne_conn.cursor()

# Create table for caching embeddings if it doesn't exist
c.execute('''
    CREATE TABLE IF NOT EXISTS embeddings (
        conversation_id TEXT PRIMARY KEY,
        embedding TEXT
    )
''')

# Create table for caching t-SNE results if it doesn't exist
tsne_c.execute('''
    CREATE TABLE IF NOT EXISTS tsne_embeddings (
        conversation_id TEXT PRIMARY KEY,
        tsne_embedding TEXT
    )
''')

# Load the dataset
dataset = load_dataset('allenai/WildChat-1M', split='train[:1000]')

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

def get_conversation_embedding(conversation_id, conversation_text):
    c.execute('SELECT embedding FROM embeddings WHERE conversation_id = ?', (conversation_id,))
    row = c.fetchone()
    if row:
        return json.loads(row[0])

    response = openai.Embedding.create(input=conversation_text, model='text-embedding-ada-002')
    embedding = response['data'][0]['embedding']

    c.execute('INSERT INTO embeddings (conversation_id, embedding) VALUES (?, ?)', (conversation_id, json.dumps(embedding)))
    conn.commit()

    return embedding

def process_conversation(conversation):
    conversation_id = conversation[0]['turn_identifier']
    conversation_text = '\n'.join([f"[{turn['role'].upper()}]: {turn['content']}" for turn in conversation])
    
    # Tokenize and truncate if necessary
    tokens = tokenizer.encode(conversation_text)
    if len(tokens) > 8192:
        tokens = tokens[:8192]
        conversation_text = tokenizer.decode(tokens)
    
    return conversation_id, conversation_text

# Generate embeddings for the first 1000 conversations
conversation_embeddings = {}
for record in tqdm(dataset):
    conversation_id, conversation_text = process_conversation(record['conversation'])
    embedding = get_conversation_embedding(conversation_id, conversation_text)
    conversation_embeddings[conversation_id] = embedding

# Save embeddings
with open('conversation_embeddings.json', 'w') as f:
    json.dump(conversation_embeddings, f)

# Load embeddings for t-SNE
embeddings = np.array(list(conversation_embeddings.values()))

# Perform t-SNE to reduce dimensions to 2D
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# Save the t-SNE model
joblib.dump(tsne, 'tsne_model.joblib')

# Save the 2D embeddings to the t-SNE database
embeddings_2d_dict = {conversation_id: embeddings_2d[i].tolist() for i, conversation_id in enumerate(conversation_embeddings.keys())}
for conversation_id, tsne_embedding in embeddings_2d_dict.items():
    tsne_c.execute('INSERT INTO tsne_embeddings (conversation_id, tsne_embedding) VALUES (?, ?)', (conversation_id, json.dumps(tsne_embedding)))
tsne_conn.commit()

# Close the database connections
conn.close()
tsne_conn.close()

print("Embeddings and t-SNE results have been processed and saved.")

