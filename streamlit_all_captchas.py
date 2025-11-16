import string
import os
import cv2
import streamlit as st
import torch
from tqdm import tqdm

from utils import data_previewer, DataObject, split_into_char_images, preprocessing_remove_lines
from model import ImageSequenceClassifier, infer, check_correctness

char_dir = 'chars_train_data/'
os.makedirs(char_dir, exist_ok=True)

st.set_page_config(layout="wide")
def main():
    if 'data' not in st.session_state:
        # Load Model
        model_path = "captcha_char_classifier_model.pth"
        net = torch.load(model_path, weights_only=False, map_location=torch.device('cpu'))

        st.session_state.data = []
        data = st.session_state.data

        wrong_count = 0
        wrong_count_with_allowance = 0
        images_count = 0
        # Loop through all png in test folder
        for (root,dirs,files) in os.walk('test/',topdown=True):
            for file in tqdm(files, desc="Loading images"):
                if file.endswith('.png'):
                    if file[1] == "_":
                        os.remove(os.path.join(root, file))
                        continue
                    if len(file.split("-")[1].split(".png")[0]) != 1: # remove any images marked wrong (if using marked train data)
                        continue
                    img_path = os.path.join(root, file)
                    img = cv2.imread(img_path)
                    processed_img = preprocessing_remove_lines(img)
                    char_images = split_into_char_images(img)
                    ground_truth = img_path.split("-")[0].split("/")[-1] 
                    
                    predictions = infer(net, img)

                    images_count += 1
                    if check_correctness(predictions, ground_truth) == False:
                        wrong_count += 1
                        
                        if check_correctness(predictions, ground_truth, with_allowance=True) == False:
                            wrong_count_with_allowance += 1
                            
                    if len(ground_truth) == len(char_images):
                        data.append(DataObject(img, processed_img, len(char_images), char_images, ground_truth, predictions))
                    continue
                    
        print(f"Images count: {images_count}")
        print(f"Wrong count: {wrong_count} - {wrong_count / images_count * 100:.2f}%")
        print(f"Wrong count with allowance: {wrong_count_with_allowance} - {wrong_count_with_allowance / images_count * 100:.2f}%")
    data_previewer(st.session_state.data)




if __name__ == "__main__":
    main()