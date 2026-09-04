import os
import requests
import gzip
import pandas as pd

# Ensure raw data directory exists
RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
os.makedirs(RAW_DIR, exist_ok=True)

def download_fssai_pdf():
    url = "https://www.fssai.gov.in/upload/uploadfiles/files/Food_Additives_Regulations.pdf"
    dest_path = os.path.join(RAW_DIR, "Food_Additives_Regulations.pdf")
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return dest_path

def download_and_process_off():
    csv_url = "https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz"
    gz_path = os.path.join(RAW_DIR, "open_food_facts.csv.gz")
    # Download gzip CSV
    with requests.get(csv_url, stream=True, timeout=60, headers={"User-Agent": "Mozilla/5.0"}) as r:
        r.raise_for_status()
        with open(gz_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    # Read CSV directly from gzip without specifying usecols (column names may differ)
    df_full = pd.read_csv(gz_path, compression="gzip", low_memory=False)
    # Define columns we need; use available ones and fill missing with NaN
    cols_of_interest = ["product_name", "ingredients_text", "nutrition_grade_fr", "nova_group"]
    df = df_full.reindex(columns=cols_of_interest)
    # Filter rows where ingredients_text is not null
    df = df[df["ingredients_text"].notna()]
    filtered_path = os.path.join(RAW_DIR, "off_filtered.csv")
    df.to_csv(filtered_path, index=False)
    return gz_path, filtered_path, len(df)

def main():
    pdf_path = download_fssai_pdf()
    gz_path, filtered_path, off_rows = download_and_process_off()
    pdf_count = 1  # we only download one PDF here
    print(f"Download summary:\n  PDFs saved: {pdf_count} ({pdf_path})\n  OFF rows saved: {off_rows} ({filtered_path})")

if __name__ == "__main__":
    main()
