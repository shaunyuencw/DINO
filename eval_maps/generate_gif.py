import os
import glob
from PIL import Image

IMAGE_FOLDER = 'images'
SET = '14-4'
IMAGE_TYPE = 'UMAP'

folders = [f.path for f in os.scandir(f"{IMAGE_FOLDER}/{SET}") if f.is_dir()]
folders.sort()
images = [f"{i}/{IMAGE_TYPE}.png" for i in folders]
for i in images:
    print(i)

def make_gif():
    frames = [Image.open(image) for image in images]
    frame_one = frames[0]
    frame_one.save(f"gifs/{SET}_{IMAGE_TYPE}.gif", format="GIF", append_images=frames,
               save_all=True, duration=len(images) * 50, loop=0)
    print(f"GIF saved as {SET}_{IMAGE_TYPE}.gif")

make_gif()