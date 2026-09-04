import io
import os
import zlib
import requests
import pandas as pd

# Ensure raw data directory exists
RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
os.makedirs(RAW_DIR, exist_ok=True)


def _decompress_partial_gzip(gz_path, chunk_size=1 << 20):
    """Decompress a gzip file that may be truncated (incomplete download).

    Uses ``zlib.decompressobj`` with ``wbits=47`` (auto-detect gzip/zlib),
    reading the compressed file in *chunk_size* byte blocks and stopping
    gracefully when a ``zlib.error`` is raised due to a missing EOF marker.

    Returns the decompressed bytes for all data that was successfully read.
    """
    # wbits=47 => 32 + 15: tells zlib to auto-detect gzip header
    d = zlib.decompressobj(wbits=47)
    out_chunks = []
    with open(gz_path, "rb") as f:
        while True:
            compressed = f.read(chunk_size)
            if not compressed:
                break
            try:
                out_chunks.append(d.decompress(compressed))
            except zlib.error:
                # Hit the truncated / corrupt section – stop here
                break
    return b"".join(out_chunks)

def download_fssai_pdf():
    url = "https://www.fssai.gov.in/upload/uploadfiles/files/Food_Additives_Regulations.pdf"
    dest_path = os.path.join(RAW_DIR, "Food_Additives_Regulations.pdf")
    if os.path.exists(dest_path):
        print(f"  [skip] {dest_path} already exists – skipping download.")
        return dest_path
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
    filtered_path = os.path.join(RAW_DIR, "off_filtered.csv")

    # ── 1. Download only if not already present ──────────────────────────────
    if os.path.exists(gz_path):
        print(f"  [skip] {gz_path} already exists – skipping download.")
    else:
        print("  Downloading Open Food Facts dataset (~1.2 GB) …")
        with requests.get(csv_url, stream=True, timeout=120,
                          headers={"User-Agent": "Mozilla/5.0"}) as r:
            r.raise_for_status()
            with open(gz_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
        print("  Download complete.")

    # ── 2. Parse – tolerates a truncated / partially-downloaded gzip ─────────
    cols_of_interest = ["product_name", "ingredients_text",
                        "nutrition_grade_fr", "nova_group"]
    print("  Parsing CSV (partial-gzip tolerant) …")
    raw_bytes = _decompress_partial_gzip(gz_path)
    print(f"  Decompressed {len(raw_bytes):,} bytes.")
    df_full = pd.read_csv(
        io.BytesIO(raw_bytes),
        sep="\t",
        low_memory=False,
        on_bad_lines="skip",
        encoding="utf-8",
        encoding_errors="replace",
    )

    # ── 3. Select & filter ────────────────────────────────────────────────────
    df = df_full.reindex(columns=cols_of_interest)
    df = df[df["ingredients_text"].notna()]
    df.to_csv(filtered_path, index=False)
    return gz_path, filtered_path, len(df)

def main():
    pdf_path = download_fssai_pdf()
    gz_path, filtered_path, off_rows = download_and_process_off()
    pdf_count = 1
    print(f"\nDownload summary:\n  PDFs saved : {pdf_count} ({pdf_path})"
          f"\n  OFF rows   : {off_rows} ({filtered_path})")

if __name__ == "__main__":
    main()
