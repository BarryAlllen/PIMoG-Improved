import os
import random
import shutil
import argparse
from PIL import Image
from tqdm import tqdm
def check_channels(image_path):
    try:
        img = Image.open(image_path)
        channels = len(img.split())
        return channels
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def select_img2dir(input, output, img_num):
    if not os.path.exists(input):
        print("Check the input dir!")
        return
    image_paths = [os.path.join(input, filename) for filename in os.listdir(input)
                   if filename.endswith(('.jpg', '.jpeg', '.png'))]

    selected_image_paths = set()
    with tqdm(total=img_num, desc="Selecting img") as pbar:
        while len(selected_image_paths) < img_num:
            img_path = random.choice(image_paths)
            channels = check_channels(img_path)
            if channels == 3 and img_path not in selected_image_paths:
                selected_image_paths.add(img_path)
                pbar.update(1)

    with tqdm(total=len(selected_image_paths), desc="Copying img") as pbar:
        for img_path in selected_image_paths:
            filename = os.path.basename(img_path)
            if not os.path.exists(output):
                os.makedirs(output)
            shutil.copy(img_path, os.path.join(output, filename))
            pbar.update(1)

def main(config):
    input_dir = config.input
    output_dir = config.output
    img_num = config.num
    select_img2dir(input_dir, output_dir, img_num)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=int, default=2)
    parser.add_argument('--input', type=str, default='/data/xujianhang/datasets/coco/train2017_64')
    parser.add_argument('--output', type=str, default='/data/xujianhang/datasets/coco/train2017_2')
    config = parser.parse_args()
    main(config)