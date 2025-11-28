# download_pexels_hd.py
import os, requests, time, csv, hashlib

# ---------- CONFIG ----------
API_KEY = "<API KEY>"   # <<-- paste your key here
OUTDIR = "pexels_group_selfies_hd"
NUM_IMAGES = 100                    # how many images you want
QUERY = "group selfie friends"
PER_PAGE = 80                       # Pexels max per_page is 80
TIMEOUT = 30
# ------------------------------

os.makedirs(OUTDIR, exist_ok=True)
meta_path = os.path.join(OUTDIR, "metadata.csv")

def safe_filename_from_url(url, prefix="pexels"):
    h = hashlib.sha1(url.encode()).hexdigest()[:10]
    return f"{prefix}_{h}.jpg"

def save_image(url, path):
    try:
        r = requests.get(url, stream=True, timeout=TIMEOUT)
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(1024*8):
                if not chunk: break
                f.write(chunk)
        return True, None
    except Exception as e:
        return False, str(e)

def fetch():
    headers = {"Authorization": API_KEY}
    saved = 0
    page = 1
    seen_urls = set()
    # write header for metadata CSV
    with open(meta_path, "w", newline="", encoding="utf-8") as mf:
        writer = csv.writer(mf)
        writer.writerow(["saved_index", "photo_id", "photographer", "photo_page_url", "src_url", "width", "height", "filename"])
    while saved < NUM_IMAGES:
        url = f"https://api.pexels.com/v1/search?query={requests.utils.quote(QUERY)}&per_page={PER_PAGE}&page={page}"
        try:
            r = requests.get(url, headers=headers, timeout=TIMEOUT)
        except Exception as e:
            print("Network error when calling Pexels API:", e)
            time.sleep(5)
            continue
        if r.status_code == 429:
            print("Rate limited (429). Backing off for 30s.")
            time.sleep(30)
            continue
        if r.status_code != 200:
            print("Pexels API returned", r.status_code, "response:", r.text[:200])
            time.sleep(5)
            continue
        data = r.json()
        photos = data.get("photos", [])
        if not photos:
            print("No photos returned on page", page, "- stopping.")
            break
        for p in photos:
            if saved >= NUM_IMAGES: break
            # Prefer 'original' (highest resolution) then 'large2x'
            src_url = p.get("src", {}).get("original") or p.get("src", {}).get("large2x")
            if not src_url: continue
            if src_url in seen_urls:
                continue
            seen_urls.add(src_url)
            fname = safe_filename_from_url(src_url, prefix="pexels")
            fpath = os.path.join(OUTDIR, fname)
            ok, err = save_image(src_url, fpath)
            if not ok:
                print("Failed to save", src_url, "error:", err)
                continue
            # record metadata
            with open(meta_path, "a", newline="", encoding="utf-8") as mf:
                writer = csv.writer(mf)
                writer.writerow([saved, p.get("id"), p.get("photographer"), p.get("url"), src_url, p.get("width"), p.get("height"), fname])
            saved += 1
            print(f"[{saved}/{NUM_IMAGES}] saved {fname}")
            time.sleep(0.15)   # be polite
        page += 1
        # small pause between pages
        time.sleep(0.5)
    print("Done. Saved", saved, "images to", OUTDIR)

if __name__ == "__main__":
    if "<YOUR_PEXELS_API_KEY>" in API_KEY or not API_KEY.strip():
        raise SystemExit("Set your PEXELS API key in API_KEY before running.")
    fetch()