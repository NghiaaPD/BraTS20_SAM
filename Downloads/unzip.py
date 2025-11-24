
import os
import gzip
import shutil

def gunzip_all_in_folder(root_folder):
	for dirpath, dirnames, filenames in os.walk(root_folder):
		for filename in filenames:
			if filename.lower().endswith('.gz'):
				gz_path = os.path.join(dirpath, filename)
				out_path = os.path.splitext(gz_path)[0]
				print(f"Extracting: {gz_path} -> {out_path}")
				with gzip.open(gz_path, 'rb') as f_in:
					with open(out_path, 'wb') as f_out:
						shutil.copyfileobj(f_in, f_out)

if __name__ == "__main__":
	# Đường dẫn tới thư mục BraTS2018
	brats2018_dir = "/home/nghiapd/medical_dataset/BraTS2018"
	gunzip_all_in_folder(brats2018_dir)
