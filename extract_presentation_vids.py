import zipfile
import os
import shutil

zip_path = r"d:\Context\Crash-1500.zip"
target_dir = r"d:\Context\presentation_videos"

os.makedirs(target_dir, exist_ok=True)

extracted_count = 0
print("Scanning Crash-1500 for presentation-worthy dashcam clips...")

try:
    with zipfile.ZipFile(zip_path, 'r') as z:
        # Get list of .mp4 files
        videos = [f for f in z.namelist() if f.endswith('.mp4') or f.endswith('.avi')]
        
        # Select first 5 videos to use as test cases
        for vid in videos[:5]:
            z.extract(vid, target_dir)
            extracted_count += 1
            print(f"Extracted: {vid}")
            
    print(f"\n✅ Successfully extracted {extracted_count} test videos into {target_dir}!")
except Exception as e:
    print(f"Error reading zip: {e}")
