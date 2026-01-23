import os
import glob
from kaggle.api.kaggle_api_extended import KaggleApi

# Configuration
# Path where data will be downloaded
DATA_DIR = os.path.join("..", "..", "data", "synthetic")
os.makedirs(DATA_DIR, exist_ok=True)

def authenticate_kaggle():
    """
    Authenticates using the ~/.kaggle/kaggle.json file.
    Ensure kaggle.json is in C:/Users/<User>/.kaggle/ or equivalent.
    """
    try:
        api = KaggleApi()
        api.authenticate()
        print("Kaggle API authenticated successfully.")
        return api
    except Exception as e:
        print(f"Authentication failed: {e}")
        print("Ensure you have placed 'kaggle.json' in your ~/.kaggle/ directory.")
        return None

def download_surrogate_data(api):
    """
    Searches for and downloads a relevant multi-omics/drug response dataset.
    """
    search_terms = ["multi omics drug response", "gene expression cancer", "CCLE"]
    
    print(f"Searching for datasets with terms: {search_terms}...")
    
    # Simple logic: try finding a specific high-quality dataset first (e.g. CCLE or TCGA)
    # The user asked to search and download the most relevant one.
    # Let's search for "CCLE" as a robust surrogate for drug response/expression if MDD is not found.
    
    found_dataset = None
    
    for term in search_terms:
        try:
            datasets = api.dataset_list(search=term, sort_by='hottest', file_type='csv')
            if datasets:
                found_dataset = datasets[0]
                print(f"Found relevant dataset: {found_dataset.ref} (Title: {found_dataset.title})")
                break
        except Exception as e:
            print(f"Error searching for {term}: {e}")
            
    if found_dataset:
        print(f"Downloading {found_dataset.ref} to {DATA_DIR}...")
        try:
            api.dataset_download_files(found_dataset.ref, path=DATA_DIR, unzip=True)
            print("Download and extraction complete.")
            
            # List downloaded files
            files = glob.glob(os.path.join(DATA_DIR, "*"))
            print(f"Files in {DATA_DIR}:")
            for f in files:
                print(f" - {os.path.basename(f)}")
                
        except Exception as e:
            print(f"Error downloading dataset: {e}")
    else:
        print("No relevant datasets found.")

if __name__ == "__main__":
    api = authenticate_kaggle()
    if api:
        download_surrogate_data(api)
