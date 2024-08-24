import os
import json
from datasets import load_dataset
from elasticsearch import Elasticsearch, helpers

# Load the dataset
dataset = load_dataset('allenai/WildChat-1M-Full')

# Initialize Elasticsearch client
#es = Elasticsearch('http://localhost:9200')
es = Elasticsearch('https://localhost:9200', basic_auth=('elastic', os.getenv('ES_PASSWD')), ssl_assert_fingerprint=os.getenv('ES_FINGERPRINT'))
#LANGUAGES = ['all', 'english', 'russian', 'chinese', 'spanish', 'german', 'french', 'portuguese', 'italian', 'japanese', 'korean']
def get_language_list():
    languages = []
    static_dir = 'static'
    
    for item in os.listdir(static_dir):
        folder_path = os.path.join(static_dir, item)
        if os.path.isdir(folder_path):
            if os.path.exists(os.path.join(folder_path, 'wildchat_embeddings.json')) and 'debug' not in item:
                languages.append(item)
    
    return languages

# Generate the LANGUAGES list
LANGUAGES = get_language_list()
#LANGUAGES = [ 'korean', 'portuguese', 'italian', 'french', 'turkish', 'german']
#LANGUAGES = [ 'english']
LANGUAGES = ['all', 'english', 'chinese', 'russian', 'spanish', 'french', 'portuguese', 'german', 'italian', 'turkish', 'arabic', 'japanese', 'korean', 'polish', 'vietnamese']
#LANGUAGES = ['english']
#LANGUAGES = ['chinese', 'russian', 'spanish']
#LANGUAGES = ['french', 'portuguese', 'german', 'italian', 'turkish']
#LANGUAGES = ['arabic', 'japanese', 'korean', 'polish', 'vietnamese']
print (LANGUAGES)

for language in LANGUAGES:
    print (language)
    index_name = f'wildchat_subset_{language}'
    mappings = {
            "properties": {
                "conversation_hash": {"type": "keyword"},
                "model": {"type": "keyword"},
                "timestamp": {"type": "date"},
                "conversation": {
                    "type": "nested",
                    "properties": {
                        "content": {"type": "text"},
                        "country": {"type": "keyword"},
                        "hashed_ip": {"type": "keyword"},
                        "header": {
                            "type": "object",
                            "properties": {
                                "accept-language": {"type": "keyword"},
                                "user-agent": {"type": "keyword"}
                            }
                        },
                        "language": {"type": "keyword"},
                        "redacted": {"type": "boolean"},
                        "role": {"type": "keyword"},
                        "state": {"type": "keyword"},
                        "timestamp": {"type": "date"},
                        "toxic": {"type": "boolean"},
                        "turn_identifier": {"type": "integer"}
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
                                "harassment_threatening": {"type": "boolean"},
                                "hate": {"type": "boolean"},
                                "hate/threatening": {"type": "boolean"},
                                "hate_threatening": {"type": "boolean"},
                                "self-harm": {"type": "boolean"},
                                "self-harm/instructions": {"type": "boolean"},
                                "self-harm/intent": {"type": "boolean"},
                                "self_harm": {"type": "boolean"},
                                "self_harm_instructions": {"type": "boolean"},
                                "self_harm_intent": {"type": "boolean"},
                                "sexual": {"type": "boolean"},
                                "sexual/minors": {"type": "boolean"},
                                "sexual_minors": {"type": "boolean"},
                                "violence": {"type": "boolean"},
                                "violence/graphic": {"type": "boolean"},
                                "violence_graphic": {"type": "boolean"}
                            }
                        },
                        "category_scores": {
                            "type": "object",
                            "properties": {
                                "harassment": {"type": "float"},
                                "harassment/threatening": {"type": "float"},
                                "harassment_threatening": {"type": "float"},
                                "hate": {"type": "float"},
                                "hate/threatening": {"type": "float"},
                                "hate_threatening": {"type": "float"},
                                "self-harm": {"type": "float"},
                                "self-harm/instructions": {"type": "float"},
                                "self-harm/intent": {"type": "float"},
                                "self_harm": {"type": "float"},
                                "self_harm_instructions": {"type": "float"},
                                "self_harm_intent": {"type": "float"},
                                "sexual": {"type": "float"},
                                "sexual/minors": {"type": "float"},
                                "sexual_minors": {"type": "float"},
                                "violence": {"type": "float"},
                                "violence/graphic": {"type": "float"},
                                "violence_graphic": {"type": "float"}
                            }
                        },
                        "flagged": {"type": "boolean"}
                    }
                },
                "detoxify_moderation": {
                    "type": "nested",
                    "properties": {
                        "identity_attack": {"type": "float"},
                        "insult": {"type": "float"},
                        "obscene": {"type": "float"},
                        "severe_toxicity": {"type": "float"},
                        "sexual_explicit": {"type": "float"},
                        "threat": {"type": "float"},
                        "toxicity": {"type": "float"}
                    }
                },
                "toxic": {"type": "boolean"},
                "redacted": {"type": "boolean"},
                "state": {"type": "keyword"},
                "country": {"type": "keyword"},
                "hashed_ip": {"type": "keyword"},
                "header": {
                    "type": "object",
                    "properties": {
                        "accept-language": {"type": "keyword"},
                        "user-agent": {"type": "keyword"}
                    }
                }
            }
    }
    
    # Create the index with the specified mappings
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
    assert not es.indices.exists(index=index_name)
    if not es.indices.exists(index=index_name):
        es.indices.create(index=index_name, mappings=mappings)
    import tqdm
    # Prepare data for bulk indexing
    actions = []
    import json
    
    wildchat_embeddings = json.load(open(f'static/{language}/wildchat_embeddings.json'))
    conversation_ids = set([item['i'] for item in wildchat_embeddings])
    chunk_size = 10000  # Set your desired chunk size
    for i, record in enumerate(tqdm.tqdm(dataset['train']), 1):
        #if i > 10000:
        #    break
        conversation_id = record['conversation'][0]['turn_identifier']
        if conversation_id not in conversation_ids:
            continue
        action = {
            "_index": index_name,
            "_source": record
        }
        actions.append(action)
        if i % chunk_size == 0:
            helpers.bulk(es, actions)
            actions = []  # Reset actions after each chunk is indexed
    
    # Index any remaining actions
    if actions:
        helpers.bulk(es, actions)
    
    print("Indexing completed.")
