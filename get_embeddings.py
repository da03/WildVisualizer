import openai
import numpy as np
from sklearn.manifold import TSNE
import json

# Replace with your API key
openai.api_key = 'YOUR_API_KEY'

def get_embedding(text):
    response = openai.Embedding.create(input=[text], model="text-embedding-ada-002")
    return response['data'][0]['embedding']

# Assume conversations is a list of conversation texts
conversations = ["conversation1", "conversation2", ...]  # Load your conversations here
embeddings = [get_embedding(convo) for convo in conversations]

# Use t-SNE to reduce dimensionality to 2D
tsne = TSNE(n_components=2, random_state=42)
reduced_embeddings = tsne.fit_transform(embeddings)

# Save the reduced embeddings to a file for later use
with open('reduced_embeddings.json', 'w') as f:
    json.dump(reduced_embeddings.tolist(), f)

