# CS4243 Mini Project: CAPTCHA Prediction with Artificial Neural Networks
### Team Members: 
Gan Ren Yick
Lee Ze Hao
Edwin Wong Jan Chung
Teo Choon Keong 

### Preprocessing
Preprocessing (line removal, character image splitting and processing) are defined in utils.py.

### Training Data
Prepare data from the "train" folder in the source data drive [https://drive.google.com/drive/folders/1JikBA_bt7HwUYge73WuohRibamdsBTcC]. Bad data can be labelled by appending "badwrong" "badcrop" or "badmiss" at the end of the file name, and will be ignored during data generation.   

Run the classifier_model_train_data_generator.ipynb notebook to generate the training data for the ANN.

Data augmentation is defined in the PyTorch Dataset later.

### Model Structure
Model structure is defined in model.py.

### Model Training
Dataset, DataLoader, training loop are defined in classifier_model_training.ipynb.

### Model Evaluation and Metrics
Final model evaluation and metrics can be performed using the classifier_model_test_eval.ipynb notebook.  
Full CAPTCHA-level and character-level metrics are provided.

### Visualization
(Activate Python environment first, run from project folder)   
Run `streamlit run streamlit_all_captchas.py` to display all captcha predictions from the test set.            
Run `streamlit run streamlit_only_wrong_captchas.py` to display all the wrongly predicted captchas from the test set.   

