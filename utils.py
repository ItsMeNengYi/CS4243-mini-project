
import streamlit as st
import numpy as np
import cv2
import os

class DataObject:
    def __init__(self, img, processed_img, ground_truth, prediction):
        self.img = img
        self.processed_img = processed_img
        self.ground_truth = ground_truth
        self.prediction = prediction

    def set_predictions(self, predictions):
        self.predictions = predictions
    

def data_previewer(data, batch_size=20):
    st.title("Data Previewer")

    if "loaded" not in st.session_state:
        st.session_state.loaded = batch_size

    items_to_show = st.session_state.loaded

    for i, item in enumerate(data[:items_to_show]):
        # Small container per data item
        with st.container():
            cols = st.columns(2)

            # Original image
            img_rgb = cv2.cvtColor(item.img, cv2.COLOR_BGR2RGB)
            cols[0].image(img_rgb, caption="Original", use_container_width=True)

            # Processed image
            processed_rgb = cv2.cvtColor(item.processed_img, cv2.COLOR_BGR2RGB)
            cols[1].image(processed_rgb, caption="Processed", use_container_width=True)

            # Compact text info
            st.markdown(
                f"**Ground Truth:** {item.ground_truth} | "
                f"**Prediction:** {item.prediction}"
            )
            st.markdown("---")  # separator

    # Load more button
    if items_to_show < len(data):
        if st.button("⬇️ Load more"):
            st.session_state.loaded += batch_size
            st.rerun()

def img_preprocessing(img):
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lines = np.where(img_grey == 0, 255, 0).astype(np.uint8)
    # Expand lines to 3 channels
    lines = cv2.merge([lines, lines, lines])
    processed = img + lines

    return processed

# Example usage
def main():

    if 'data' not in st.session_state:
        st.session_state.data = []
        data = st.session_state.data

        # Loop through all png in train/
        for (root,dirs,files) in os.walk('train/',topdown=True):
            for file in files:
                if file.endswith('.png'):
                    img_path = os.path.join(root, file)
                    img = cv2.imread(img_path)
                    processed_img = img_preprocessing(img)
                    ground_truth = img_path.split("-")[0].split("/")[-1] 
                    prediction = "Example Prediction"  # Placeholder
                    data.append(DataObject(img, processed_img, ground_truth, prediction))

    data_previewer(st.session_state.data)

if __name__ == "__main__":
    main()