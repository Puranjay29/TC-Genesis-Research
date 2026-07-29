import os
import glob
import shutil
import argparse

def main():
    parser = argparse.ArgumentParser(description="Split balanced patches temporally")
    parser.add_argument("--indir", default="./data/binary_classification_dataset", help="Input dataset path")
    parser.add_argument("--outdir", default="./data/split_dataset", help="Output directory path")
    parser.add_argument("--val-from", default="20200101", help="Validation split start YYYYMMDD")
    parser.add_argument("--test-from", default="20210101", help="Test split start YYYYMMDD")
    args = parser.parse_args()

    for c in ['pos', 'neg']:
        search_path = os.path.join(args.indir, c, "*.nc")
        files = glob.glob(search_path)
        print(f"Processing class: {c} ({len(files)} total files)")

        for f in files:
            fname = os.path.basename(f)
            # Extracted date string from filename start (e.g. '20080505')
            date_str = fname.split('_')[0]

            if date_str >= args.test_from:
                split_type = "test"
            elif date_str >= args.val_from:
                split_type = "val"
            else:
                split_type = "train"

            target_dir = os.path.join(args.outdir, f"dataset_{split_type}", c)
            os.makedirs(target_dir, exist_ok=True)
            shutil.copy(f, os.path.join(target_dir, fname))

    print("Success! Tensors partitioned chronologically.")

if __name__ == "__main__":
    main()
