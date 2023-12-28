import os
from tqdm.auto import tqdm

def write_filenames_to_txt(folder_path, output):
    output = os.path.join(folder_path, output)
    if os.path.exists(output):
        os.remove(output)

    file_names = os.listdir(folder_path)
    total_files = len(file_names)
    with open(output, 'w') as txt_file:
        with tqdm(total=total_files, desc="Writing Files") as pbar:
            for filename in file_names:
                txt_file.write(f"{filename} {filename}\n")
                pbar.update(1)