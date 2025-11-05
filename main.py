import string
import os
import cv2
import streamlit as st
import torch
from tqdm import tqdm

from utils import data_previewer, DataObject, preprocessing_remove_lines, preprocessing_split_by_characters, process_character_image
from model import Model, infer, check_correctness

keys = list(string.ascii_lowercase) + [str(i) for i in range(10)]
char_dict = {k: [] for k in keys}
char_dir = 'chars_train_data/'
os.makedirs(char_dir, exist_ok=True)

st.set_page_config(layout="wide")
def main():
    if 'data' not in st.session_state:
        # --- 1. Load Model ---
        model_path = "char_classification_model.pth"
        net = torch.load(model_path, weights_only=False)

        st.session_state.data = []
        data = st.session_state.data

        wrong_count = 0
        images_count = 0
        # Loop through all png in train/
        for (root,dirs,files) in os.walk('test/',topdown=True):
            for file in tqdm(files, desc="Loading images"):
                if file.endswith('.png'):
                    if file[1] == "_":
                        os.remove(os.path.join(root, file))
                        continue
                    if file.split("-")[1].split(".png")[0] != "0":
                        continue
                    img_path = os.path.join(root, file)
                    img = cv2.imread(img_path)
                    processed_img = preprocessing_remove_lines(img)
                    num_chars, char_images = preprocessing_split_by_characters(processed_img)
                    char_images = [process_character_image(ci) for ci in char_images]
                    ground_truth = img_path.split("-")[0].split("/")[-1] 
                    
                    predictions = infer(net, img)

                    images_count += 1
                    if check_correctness(predictions, ground_truth) == False:
                        wrong_count += 1
                        if len(ground_truth) == len(char_images):
                            data.append(DataObject(img, processed_img, num_chars, char_images, ground_truth, predictions))
                        continue

                    # Correct separated
                    for i in range(len(ground_truth)):
                        k = ground_truth[i]
                        char_dict[k].append(char_images[i])
         # --- 4. Save each image to PNG ---
        for k, img_list in char_dict.items():
            for idx, img in enumerate(img_list):
                filename = f"{k}_{idx:05d}.png"  # e.g., a_0000.png
                path = os.path.join(char_dir, filename)
                cv2.imwrite(path, img)
        print(f"Wrong count: {wrong_count}")
    data_previewer(st.session_state.data)




if __name__ == "__main__":
    main()