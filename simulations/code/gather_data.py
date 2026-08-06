import os
import shutil

def collect_dust_fits_files():
    # Create the destination directory if it doesn't exist
    dest_dir = "data"
    os.makedirs(dest_dir, exist_ok=True)

    # Loop through all directories starting with 'x'
    for d in os.listdir('.'):
        if os.path.isdir(d) and d.startswith('x'):
            fits_dir = os.path.join(d, 'fits_images')
            if os.path.isdir(fits_dir):
                # Iterate through files inside fits_images/
                for file_name in os.listdir(fits_dir):
                    if file_name.startswith('dust') and file_name.endswith('.fits'):
                        src_path = os.path.join(fits_dir, file_name)
                        dst_path = os.path.join(dest_dir, file_name)

                        # If the file already exists, rename it to avoid overwrite
                        if os.path.exists(dst_path):
                            base, ext = os.path.splitext(file_name)
                            counter = 1
                            while os.path.exists(dst_path):
                                dst_path = os.path.join(dest_dir, f"{base}_{counter}{ext}")
                                counter += 1

                        # Copy the file
                        shutil.copy2(src_path, dst_path)
                        print(f"Copied: {src_path} → {dst_path}")

    print(f"\n✅ All matching FITS files have been copied to '{dest_dir}'.")

if __name__ == "__main__":
    collect_dust_fits_files()