import os
import GEOparse
import pandas as pd
from Bio import Entrez

# Configuration
DATA_DIR = os.path.join("..", "..", "data", "raw", "transcriptomics")
os.makedirs(DATA_DIR, exist_ok=True)

def download_specific_dataset(gse_id="GSE98793"):
    """
    Downloads a specific GEO dataset (GSE) and saves it to the data directory.
    """
    print(f"Attempting to download {gse_id}...")
    try:
        gse = GEOparse.get_GEO(geo=gse_id, destdir=DATA_DIR)
        print(f"Successfully downloaded {gse_id}")
        
        # Save metadata to a simple text file for quick reference
        metadata_file = os.path.join(DATA_DIR, f"{gse_id}_metadata.txt")
        with open(metadata_file, "w") as f:
            f.write(str(gse.metadata))
        print(f"Metadata saved to {metadata_file}")
        
    except Exception as e:
        print(f"Error downloading {gse_id}: {e}")

def search_geo_metadata(email, term="Antidepressant Response AND MDD Gene Expression"):
    """
    Searches NCBI GEO for datasets matching the term and saves the results.
    Requires an email for Entrez usage.
    """
    print(f"Searching GEO for: {term}")
    Entrez.email = email
    
    try:
        # Search for datasets (GDS) or Series (GSE)
        handle = Entrez.esearch(db="gds", term=term, retmax=20)
        results = Entrez.read(handle)
        handle.close()
        
        id_list = results['IdList']
        print(f"Found {len(id_list)} results.")
        
        if not id_list:
            return

        # Fetch details for found IDs
        handle = Entrez.esummary(db="gds", id=",".join(id_list))
        summaries = Entrez.read(handle)
        handle.close()
        
        # Parse and save summaries
        search_results = []
        for summary in summaries:
            search_results.append({
                "Accession": summary.get("Accession"),
                "Title": summary.get("Title"),
                "Summary": summary.get("Summary"),
                "PDAT": summary.get("PDAT") # Publication Date
            })
            
        df = pd.DataFrame(search_results)
        output_path = os.path.join(DATA_DIR, "geo_search_results.csv")
        df.to_csv(output_path, index=False)
        print(f"Search results saved to {output_path}")
        print(df.head())

    except Exception as e:
        print(f"Error during metadata search: {e}")

if __name__ == "__main__":
    # 1. Download the specific MDD Gene Expression dataset
    download_specific_dataset("GSE98793")

    # 2. Search for related datasets
    # REPLACE WITH YOUR ACTUAL EMAIL
    user_email = "your_email@example.com" 
    search_geo_metadata(user_email)
