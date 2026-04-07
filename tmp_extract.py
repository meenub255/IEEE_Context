import zipfile
import os

def extract_archive(zip_path, dest_dir):
    print(f"Extracting {zip_path} to {dest_dir}...")
    os.makedirs(dest_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest_dir)
    print(f"Extraction of {zip_path} complete.")

if __name__ == '__main__':
    base_dir = r"d:\Context"
    extract_archive(os.path.join(base_dir, "acdc-computer_vision.zip"), os.path.join(base_dir, "dataset", "acdc"))
    extract_archive(os.path.join(base_dir, "exdark.zip"), os.path.join(base_dir, "dataset", "exdark"))
