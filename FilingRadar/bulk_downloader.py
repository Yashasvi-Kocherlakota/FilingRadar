# bulk_downloader.py

import requests
import os
from asset_managers import asset_managers
from bs4 import BeautifulSoup

HEADERS = {'User-Agent': 'FilingRadarBot (your_email@example.com)'}

def get_filing_url(cik, form_type="10-K"):
    cik_formatted = cik.zfill(10)
    base_url = f"https://data.sec.gov/submissions/CIK{cik_formatted}.json"
    res = requests.get(base_url, headers=HEADERS)
    if res.status_code != 200:
        print(f"[ERROR] Failed to get submissions for CIK {cik}")
        return None

    data = res.json()
    filings = data.get("filings", {}).get("recent", {})
    for i, f_type in enumerate(filings.get("form", [])):
        if f_type == form_type:
            accession_raw = filings["accessionNumber"][i]
            accession_clean = accession_raw.replace("-", "")
            doc_base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_clean}/"
            index_url = doc_base + "index.json"
            index_res = requests.get(index_url, headers=HEADERS)
            if index_res.ok:
                index_data = index_res.json()
                for file in index_data.get("directory", {}).get("item", []):
                    if file["name"].endswith(".htm") and "10-k" in file["name"].lower():
                        return doc_base + file["name"]
    return None

def download_and_save_filing(cik, name, form_type="10-K"):
    url = get_filing_url(cik, form_type)
    if not url:
        print(f"[SKIP] {name}: No recent {form_type} found.")
        return

    res = requests.get(url, headers=HEADERS)
    if res.status_code != 200:
        print(f"[ERROR] Failed to download filing for {name}")
        return

    soup = BeautifulSoup(res.text, "html.parser")
    text = soup.get_text(separator=" ")

    os.makedirs("filings", exist_ok=True)
    filename = f"filings/{name.replace(' ', '_').lower()}_{form_type.lower()}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"[SAVED] {name}: {form_type} saved to {filename}")

if __name__ == "__main__":
    for company in asset_managers:
        download_and_save_filing(company["cik"], company["name"], "10-K")
