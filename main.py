import string
import os
import cv2
import streamlit as st
from tqdm import tqdm

from utils import data_previewer, DataObject, preprocessing_remove_lines, preprocessing_split_by_characters

keys = list(string.ascii_lowercase) + [str(i) for i in range(10)]
char_dict = {k: [] for k in keys}
char_dir = 'chars_train_data/'
os.makedirs(char_dir, exist_ok=True)
def main():
    if 'data' not in st.session_state:
        st.session_state.data = []
        data = st.session_state.data

        wrong_count = 0
        images_count = 0
        # Loop through all png in train/
        for (root,dirs,files) in os.walk('clean_train_data/',topdown=True):
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
                    ground_truth = img_path.split("-")[0].split("/")[-1] 
                    prediction = "Example Prediction"  # Placeholder

                    images_count += 1
                    if len(char_images) != len(ground_truth):
                        wrong_count += 1
                        data.append(DataObject(img, processed_img, num_chars, char_images, ground_truth, prediction))
                        continue

        #             # Correct seperated
        #             for i in range(len(ground_truth)):
        #                 k = ground_truth[i]
        #                 char_dict[k].append(char_images[i])
        # # --- 4. Save each image to PNG ---
        # for k, img_list in char_dict.items():
        #     for idx, img in enumerate(img_list):
        #         filename = f"{k}_{idx:05d}.png"  # e.g., a_0000.png
        #         path = os.path.join(char_dir, filename)
        #         cv2.imwrite(path, img)
        print(f"Wrong count: {wrong_count}")
    data_previewer(st.session_state.data)




if __name__ == "__main__":
    main()