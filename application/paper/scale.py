# %%
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def depth_read(filename):
    depth_png = np.array(Image.open(filename), dtype=int)
    assert(np.max(depth_png) > 255)
    depth = depth_png.astype(np.float) / 256.
    depth[depth_png == 0] = -1.
    return depth

def process_file(file_path):
    try:
        depth = depth_read(file_path)
        return depth[depth != -1].flatten()
    except AssertionError:
        return []

# Set the directory path
directory_path = '/data1/Chenbingyuan/Trans_G2/g2_dataset/scale/train'
pattern = f"{directory_path}/**/*groundtruth/image_02/**/*.png"
file_paths = glob.glob(pattern, recursive=True)

depth_values = []

with ThreadPoolExecutor() as executor:
    futures = [executor.submit(process_file, file_path) for file_path in file_paths]
    for future in tqdm(as_completed(futures), total=len(futures)):
        depth_values.extend(future.result())

depth_values_np = np.array(depth_values)

if depth_values_np.size > 0:
    min_depth = np.min(depth_values_np)
    max_depth = np.max(depth_values_np)
    mean_depth = np.mean(depth_values_np)
    std_depth = np.std(depth_values_np)
else:
    min_depth, max_depth, mean_depth, std_depth = (None, None, None, None)

print(f'min_depth={min_depth}, max_depth={max_depth}, mean_depth={mean_depth}, std_depth={std_depth}')


# %%



