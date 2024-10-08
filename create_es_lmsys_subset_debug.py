import os
import json
from datasets import load_dataset
from elasticsearch import Elasticsearch, helpers

# Initialize Elasticsearch client (same as before)
es = Elasticsearch('https://localhost:9200', basic_auth=('elastic', os.getenv('ES_PASSWD')), ssl_assert_fingerprint=os.getenv('ES_FINGERPRINT'))


LANGUAGES = ['all', 'english', 'russian', 'chinese', 'spanish', 'german', 'french', 'portuguese', 'italian', 'japanese', 'korean']
LANGUAGES = ['english']
LANGUAGES = ['all']

for language in LANGUAGES:
    print (language)
    lmsys_index_name = f'lmsyschat_subset_{language}'
    lmsys_mappings = {
        "properties": {
            "conversation_id": {"type": "keyword"},
            "model": {"type": "keyword"},
            "conversation": {
                "type": "nested",
                "properties": {
                    "content": {"type": "text"},
                    "role": {"type": "keyword"},
                }
            },
            "turn": {"type": "integer"},
            "language": {"type": "keyword"},
            "openai_moderation": {
                "type": "nested",
                "properties": {
                    "categories": {
                        "type": "object",
                        "properties": {
                            "harassment": {"type": "boolean"},
                            "harassment/threatening": {"type": "boolean"},
                            "hate": {"type": "boolean"},
                            "hate/threatening": {"type": "boolean"},
                            "self-harm": {"type": "boolean"},
                            "self-harm/instructions": {"type": "boolean"},
                            "self-harm/intent": {"type": "boolean"},
                            "sexual": {"type": "boolean"},
                            "sexual/minors": {"type": "boolean"},
                            "violence": {"type": "boolean"},
                            "violence/graphic": {"type": "boolean"}
                        }
                    },
                    "category_scores": {
                        "type": "object",
                        "properties": {
                            "harassment": {"type": "float"},
                            "harassment/threatening": {"type": "float"},
                            "hate": {"type": "float"},
                            "hate/threatening": {"type": "float"},
                            "self-harm": {"type": "float"},
                            "self-harm/instructions": {"type": "float"},
                            "self-harm/intent": {"type": "float"},
                            "sexual": {"type": "float"},
                            "sexual/minors": {"type": "float"},
                            "violence": {"type": "float"},
                            "violence/graphic": {"type": "float"}
                        }
                    },
                    "flagged": {"type": "boolean"}
                }
            },
            "redacted": {"type": "boolean"}
        }
    }
    
    # Create the LMSYS-Chat-1M index
    if es.indices.exists(index=lmsys_index_name):
        es.indices.delete(index=lmsys_index_name)
    assert not es.indices.exists(index=lmsys_index_name)
    if not es.indices.exists(index=lmsys_index_name):
        es.indices.create(index=lmsys_index_name, mappings=lmsys_mappings)
    
    # Load and index the LMSYS-Chat-1M dataset similarly to how you did for WildChat
    dataset = load_dataset('lmsys/LMSYS-Chat-1M')
    
    import tqdm
    # Prepare data for bulk indexing
    actions = []
    import json
    
    lmsyschat_embeddings = json.load(open(f'static/{language}/lmsyschat_embeddings.json'))
    conversation_ids = set([item['i'] for item in lmsyschat_embeddings])
    chunk_size = 10000  # Set your desired chunk size
    for i, record in enumerate(tqdm.tqdm(dataset['train']), 1):
        #if i > 10000:
        #    break
        conversation_id = record['conversation_id']
        if conversation_id not in conversation_ids:
            continue
        action = {
            "_index": lmsys_index_name,
            "_source": record
        }
        actions.append(action)
        if i % chunk_size == 0:
            helpers.bulk(es, actions)
            actions = []  # Reset actions after each chunk is indexed
    
    # Index any remaining actions
    if actions:
        helpers.bulk(es, actions)
    
    print(f"Indexing completed for {language}.")
print("LMSYS-Chat-1M Indexing completed.")
