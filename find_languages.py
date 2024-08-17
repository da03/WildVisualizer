from datasets import load_dataset
import pandas as pd

def get_language_distribution(dataset_name):
    # Load the dataset
    dataset = load_dataset(dataset_name, split='train')
    
    # Assuming 'language' field is directly under dataset, adjust if nested
    # Collect language data
    language_data = dataset['language']
    
    # Calculate language frequency
    language_counts = pd.Series(language_data).value_counts()
    
    return language_counts

def main():
    # Dataset names, adjust if they are different
    wildchat_dataset = 'allenai/WildChat-1M-Full'
    lmsys_dataset = 'lmsys/LMSYS-Chat-1M'
    
    # Get language distributions
    wildchat_languages = get_language_distribution(wildchat_dataset)
    lmsys_languages = get_language_distribution(lmsys_dataset)
    
    # Combine the counts using min function for each language
    combined_min_counts = pd.concat([wildchat_languages, lmsys_languages], axis=1, keys=['WildChat', 'LMSYS']).min(axis=1)
    
    # Sort the counts from highest to lowest
    sorted_min_counts = combined_min_counts.sort_values(ascending=False)
    pd.set_option('display.max_rows', None)

    
    print(sorted_min_counts)

if __name__ == "__main__":
    main()

