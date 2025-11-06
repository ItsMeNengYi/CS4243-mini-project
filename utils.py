
import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import uniform_filter

class DataObject:
    def __init__(self, img, processed_img, number_of_characters, list_of_character_images, ground_truth, predictions):
        self.img = img
        self.processed_img = processed_img
        self.ground_truth = ground_truth
        self.predictions = predictions
        self.number_of_characters = number_of_characters
        self.character_images = list_of_character_images

    def set_predictions(self, predictions):
        self.predictions = predictions
    

def data_previewer(data, batch_size=30):
    if "loaded" not in st.session_state:
        st.session_state.loaded = batch_size

    items_to_show = st.session_state.loaded

    for i, item in enumerate(data[:items_to_show]):
        # Small container per data item
        with st.container():
            cols = st.columns(2)

            # Original image
            img_rgb = cv2.cvtColor(item.img, cv2.COLOR_BGR2RGB)
            cols[0].image(img_rgb, caption="Original", width='content')

            # Processed image
            processed_rgb = cv2.cvtColor(item.processed_img, cv2.COLOR_BGR2RGB)
            cols[1].image(processed_rgb, caption="Processed", width='content')


            cols2 = st.columns(len(item.character_images))
            for j, char_img in enumerate(item.character_images):
                char_rgb = cv2.cvtColor(char_img, cv2.COLOR_BGR2RGB)
                cols2[j].image(char_rgb, caption=f"Truth: {item.ground_truth[j] if j < len(item.ground_truth) else 'N/A'} | Pred: {item.predictions[j] if j < len(item.predictions) else 'N/A'}")

            # Compact text info
            pred_string = "".join(item.predictions)
            st.markdown(
                f"**Ground Truth:** {item.ground_truth} | **Prediction:** {pred_string}"
            )
            st.markdown("---")  # separator

    # Load more button
    if items_to_show < len(data):
        if st.button("⬇️ Load more"):
            st.session_state.loaded += batch_size
            st.rerun()


def preprocessing_remove_lines(img) -> np.ndarray: 
    
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lines = np.where(img_grey == 0, 255, 0).astype(np.uint8)
    # Expand lines to 3 channels
    lines = cv2.merge([lines, lines, lines])
    processed = np.clip(img + lines, 0, 255).astype(np.uint8)

    mask_black = np.all(img == 0, axis=-1)
    
    # Compute 3x3 minimum for each channel
    # kernel = np.ones((3, 3), np.uint8) 
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)) # cross-shaped kernel, no corners
    min_img = np.stack([cv2.erode(processed[..., c], kernel) for c in range(3)], axis=-1)

    # Replace black pixels with neighborhood minima
    result = img.copy()
    result[mask_black] = min_img[mask_black]
    
    """
    # --- CV2 INPAINT METHOD ---
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Create a mask for black pixels (since lines are black)
    # Adjust threshold if lines aren't perfectly black
    mask = cv2.inRange(gray, 0, 1)
    
    # Inpaint (remove black lines and fill with nearby pixels)
    result = cv2.inpaint(img, mask, inpaintRadius=1, flags=cv2.INPAINT_TELEA)

    # Create a mask for pixels that are nearly white across all channels
    near_white_mask = np.all(result > 240, axis=-1)
    
    # Set those pixels to pure white
    result[near_white_mask] = [255, 255, 255]
    """

    return result

def show_img(img: np.ndarray, title=""):
    # If image has 2 dimensions → grayscale
    if len(img.shape) == 2:
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    # If image has 3 dimensions → color
    elif len(img.shape) == 3:
        # Convert from OpenCV BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)

    plt.title(title)
    plt.show()

def show_imgs(img_list: list[np.ndarray], title=""):
    num_images = len(img_list)
    # Create 1 row, N columns
    fig, axes = plt.subplots(1, num_images, figsize=(num_images*3, 3))  # adjust width per image

    # Ensure axes is always a list
    if num_images == 1:
        axes = [axes]

    for ax, img in zip(axes, img_list):
        # If image has 2 dimensions → grayscale
        if len(img.shape) == 2:
            ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        # If image has 3 dimensions → color
        elif len(img.shape) == 3:
            # Convert from OpenCV BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
    
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_graph(arr):
    plt.figure(figsize=(12, 4))
    plt.plot(arr)
    plt.show()

def am_power(a):
    dim1 = a.shape
    
    if len(dim1)==2:
        sz = dim1[0] * dim1[1] 
    elif len(dim1)==3:
        sz = dim1[0] * dim1[1] * dim1[2]        
    pa = np.sum( a ** 2.0) / sz    
    
    return pa

def preprocessing_split_by_characters(img: np.ndarray) -> tuple[int, list[np.ndarray]]:
    grey = np.mean(img, axis=2)
    grey_no_white = np.where(grey == 255, 0, 1)
    if am_power(grey_no_white) < 0.02:
        return 0, split_by_column(img)[1]
    else:
        return 1, segment_by_color(img)[1]

def split_by_column(img: np.ndarray) -> tuple[int, list[np.ndarray]]:
    grey_scale = img.mean(axis=2)
    grey_scale = cv2.GaussianBlur(grey_scale, (5,5), 0)
    
    col_values = np.where(grey_scale == 255, 0, 1)
    col_values = np.sum(col_values, axis=0)
    
    find_left = True
    unique_cols_range = []
    left = 0
    for i in range(len(col_values) - 1):
        if find_left and col_values[i] == 0 and col_values[i + 1] > 0:
            left = i
            find_left = False
        elif not find_left and col_values[i] > 0 and col_values[i + 1] == 0:
            right = i + 1
            unique_cols_range.append((left, right))
            find_left = True
    
    char_images = []
    for r in unique_cols_range:
        char_img = np.ones_like(img) * 255
        char_img[:, r[0]:r[1], :] = img[:, r[0]:r[1], :]
        char_images.append(char_img)

    return len(char_images), char_images
def segment_by_color(
    img: np.ndarray, 
    eps: float = 0.5,           # DBSCAN neighborhood radius (tuned)
    min_pixel_area: int = 31,   # Minimum pixel count per final blob
    db_min_samples: int = 27,   # Minimum pixels to form a cluster
    spatial_weight: float = 0.1  # Controls importance of x,y distance
) -> tuple[int, list[np.ndarray]]:
    """
    Segments a CAPTCHA image using spatial + color (hue) clustering.

    - Converts image to HSV and filters out white background.
    - Maps hue to (cos, sin) to handle circular hue wraparound.
    - Clusters in 4D space: (x, y, cos(hue), sin(hue)) using DBSCAN.
    - Merges small gaps vertically to connect 'i' stems/dots.
    
    Args:
        img: Input RGB (or BGR) image as NumPy array.
        eps: DBSCAN neighborhood size (after scaling).
        min_pixel_area: Minimum size for a valid character blob.
        db_min_samples: Minimum samples for DBSCAN cluster.
        spatial_weight: Relative importance of spatial vs color distance.
    
    Returns:
        (character_count, list_of_character_images)
    """
    # --- 1. Convert to HSV ---
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]

    # --- 2. Filter out white background ---
    s_thresh = 5
    v_thresh = 250
    bg_mask = cv2.bitwise_and(
        cv2.compare(s_channel, s_thresh, cv2.CMP_LT),
        cv2.compare(v_channel, v_thresh, cv2.CMP_GT)
    )
    fg_mask = cv2.bitwise_not(bg_mask)

    y_coords, x_coords = np.where(fg_mask > 0)
    if len(y_coords) < min_pixel_area:
        return 0, []

    # --- 3. Get hue values and map to circular coords ---
    hues = hsv[y_coords, x_coords, 0].astype(np.float32) / 180.0  # normalize hue to [0,1]
    hue_x = np.cos(2 * np.pi * hues)
    hue_y = np.sin(2 * np.pi * hues)

    # --- 4. Combine features: spatial + color ---
    features = np.column_stack([
        y_coords, 
        x_coords,
        hue_x,
        hue_y
    ])

    # --- 5. Normalize features (to make x,y comparable to hue) ---
    features_scaled = StandardScaler().fit_transform(features)
    features_scaled[:, :2] *= spatial_weight  # re-apply spatial weight after scaling

    # --- 6. DBSCAN clustering ---
    db = DBSCAN(eps=eps, min_samples=db_min_samples).fit(features_scaled)
    labels = db.labels_
    unique_labels = set(labels)

    char_images = []
    char_bboxes = [[],[]]

    # --- 7. Build masks for each cluster ---
    for label in unique_labels:
        if label == -1:
            continue  # skip noise

        cluster_mask = np.zeros(img.shape[:2], dtype="uint8")
        cluster_idx = np.where(labels == label)[0]
        cluster_y = y_coords[cluster_idx]
        cluster_x = x_coords[cluster_idx]
        cluster_mask[cluster_y, cluster_x] = 255

        # --- 8. Morphological close (vertical kernel) ---
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
        closed_mask = cv2.morphologyEx(cluster_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # --- 9. Find connected components (characters) ---
        num_comp, comp_labels, stats, _ = cv2.connectedComponentsWithStats(closed_mask, 8, cv2.CV_32S)
        if num_comp <= 1:
            continue

        for i in range(1, num_comp):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_pixel_area:
                continue

            final_mask = (comp_labels == i)
            y_pix, x_pix = np.where(final_mask)
            if len(x_pix) == 0:
                continue

            x_min = x_pix.min()
            x_max = x_pix.max()

            # Create character image
            char_img = np.full_like(img, 255, dtype=np.uint8)
            char_img[final_mask] = img[final_mask]

            char_images.append(char_img)
            char_bboxes[0].append(x_min)
            char_bboxes[1].append(x_max)
        

    # --- 10. Sort characters left-to-right ---
    if char_bboxes:
        sorted_idx = np.argsort(char_bboxes[0])
        char_images = [char_images[i] for i in sorted_idx]
        char_bboxes[0] = [char_bboxes[0][i] for i in sorted_idx]
        char_bboxes[1]= [char_bboxes[1][i] for i in sorted_idx]

    # --- 11. Merge similar hue value and neighbour imgs
    hue_diff_thres = 0.5
    fin_images = []
    i = 0
    while( i < len(char_images)):
        if i == len(char_images) - 1:
            fin_images.append(char_images[i])
            break
        min_i, max_i = char_bboxes[0][i], char_bboxes[1][i]
        min_j, max_j = char_bboxes[0][i + 1], char_bboxes[1][i + 1]
        mean_i = (min_i + max_i) / 2
        mean_j = (min_j + max_j) / 2
        is_contain = (min_i < mean_j and max_i > mean_j) and (min_j < mean_i and max_j > mean_i)
        
        if is_contain and abs(mean_hue(char_images[i])- mean_hue(char_images[i+1])) < hue_diff_thres:
            fin_images.append(char_images[i] + char_images[i+1])
            i += 2
        else:
            fin_images.append(char_images[i])
            i += 1

    return len(fin_images), fin_images


def mean_hue(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    grey = np.mean(img,axis=2)
    return np.sum(np.where(grey!=255, hsv[:,:,0],0)) / np.sum(np.where(grey!=255, 1,0))

def process_character_image(img):
    # --- Convert to grayscale ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # --- Invert so character = white (255), background = black (0) ---
    gray = 255 - gray
    
    # --- Threshold to make binary ---
    #_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # --- Find bounding box of character ---
    coords = cv2.findNonZero(gray)
    x, y, w, h = cv2.boundingRect(coords)
    
    # --- Crop + 3px padding ---
    pad = 3
    x1 = max(x - pad, 0)
    y1 = max(y - pad, 0)
    x2 = min(x + w + pad, gray.shape[1])
    y2 = min(y + h + pad, gray.shape[0])
    cropped = gray[y1:y2, x1:x2]

    # --- Normalize brightness: stretch intensity range to full 0–255 ---
    min_val, max_val = np.min(cropped), np.max(cropped)
    if max_val > min_val:  # avoid divide-by-zero if image is uniform
        cropped = (cropped - min_val) * (255.0 / (max_val - min_val))
        cropped = np.clip(cropped, 0, 255).astype(np.uint8)
    
    # --- Target canvas and padding ---
    target_size = 42
    pad = 3
    available_size = target_size - 2 * pad  # 38×38 drawable area

    h, w = cropped.shape
    scale = min(available_size / h, available_size / w)  # scale to fit within 38×38

    # --- Resize with preserved aspect ratio ---
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)

    # --- Create blank 42×42 black background ---
    canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    
    # --- Center the character ---
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return canvas

def split_into_char_images(img):
    img = preprocessing_remove_lines(img) # Remove lines
    _, char_images = preprocessing_split_by_characters(img) # Split chars
    processed_char_images = []
    for char_img in char_images:
        processed_char_images.append(process_character_image(char_img)) 
    return processed_char_images

# ------DEPRECATED FUNCTIONS------
# def emphasize_spikes(signal, w=11):
#     """
#     Emphasizes main spikes by comparing the middle value to its neighbors.
#     Smaller nearby spikes are suppressed.
#     """
#     # Make sure window size is odd
#     if w % 2 == 0:
#         w += 1

#     # Build kernel: middle = 1, others = -1/(w-1)
#     kernel = np.ones(w) * (-1/ (w - 1))
#     kernel[w // 2] = 1

#     # Convolve
#     enhanced = convolve1d(signal, kernel, mode='reflect')
#     enhanced = np.clip(enhanced, 0, None)
#     return enhanced

# def clean_noise(imgs):
#     max_pixels_count = 0
#     imgs_char_ranges = []
#     for img in imgs:
#         img = np.mean(img,axis=2)
#         col_values = np.sum(img, axis=0)

#         left = True
#         prev = None
#         char_ranges = []
#         for i in range(len(col_values) - 1):
#             if left and (col_values[i] == 0 and col_values[i + 1] > 0):
#                 prev = i
#                 left = False
#             elif not left and (col_values[i] > 0 and col_values[i + 1] == 0):
#                 char_ranges.append([prev, i + 1])
#                 left = True
#                 max_pixels_count = max(max_pixels_count, np.sum(np.where(img[:, prev:i + 1] != 0, 1, 0)))
#         imgs_char_ranges.append(char_ranges)
        
#     fin_images = []
#     for i in range(len(imgs_char_ranges)):
#         if len(imgs_char_ranges[i]) == 1:
#             fin_images.append(imgs[i])
#             continue
#         chars_ranges = imgs_char_ranges[i]
#         # Create new images based on the true_chars_col_range
#         for start, end in chars_ranges:
#             pixel_count = np.sum(np.where(np.mean(imgs[i][:, start:end + 1, :], axis=2) != 0, 1, 0))
#             if pixel_count < max_pixels_count / 5:
#                 # show_img(imgs[i][:, start:end + 1, :], title="Filtered noise")
#                 continue
#             # Create a black background image
#             char_img = np.zeros_like(imgs[i])
#             char_img[:, start:end + 1, :] = imgs[i][:, start:end + 1, :]
#             fin_images.append(char_img)

#     return fin_images