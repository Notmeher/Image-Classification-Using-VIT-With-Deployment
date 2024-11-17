import streamlit as st
from transformers import AutoConfig, AutoModelForImageClassification, AutoFeatureExtractor
import torch
from PIL import Image
import os

# Define the folder where your model and config files are saved
folder_name = "D:/Random/FInal_code/flo/vit"

# Load the model configuration using the folder path
try:
    config = AutoConfig.from_pretrained(folder_name)
    st.write("Configuration loaded successfully.")
except Exception as e:
    st.error(f"Failed to load configuration from {folder_name}: {e}")
    st.stop()

# Load the preprocessor configuration (if needed for feature extraction)
try:
    preprocessor_config = AutoFeatureExtractor.from_pretrained(folder_name)
    st.write("Preprocessor configuration loaded successfully.")
except Exception as e:
    preprocessor_config = None
    st.warning(f"Preprocessor configuration not found or not needed: {e}")

# Load the model weights using the folder path
try:
    model = AutoModelForImageClassification.from_pretrained(folder_name, config=config)
    st.write("Model loaded successfully.")
except Exception as e:
    st.error(f"Failed to load model from {folder_name}: {e}")
    st.stop()

# Define mapping for labels
label_map = {
    0: 'calendula', 1: 'coreopsis', 2: 'rose', 3: 'black_eyed_susan', 4: 'water_lily',
    5: 'california_poppy', 6: 'dandelion', 7: 'magnolia', 8: 'astilbe', 9: 'sunflower',
    10: 'tulip', 11: 'bellflower', 12: 'iris', 13: 'common_daisy', 14: 'daffodil', 15: 'carnation'
}

# Define function to predict label
def predict_label(img):
    # Preprocess the image using the preprocessor if available
    if preprocessor_config:
        img_tensor = preprocessor_config(images=img, return_tensors="pt").pixel_values
    else:
        # Standard image preprocessing if no preprocessor config
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img_tensor = transform(img).unsqueeze(0)

    # Perform inference
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        outputs = model(img_tensor)
        logits = outputs.logits

    # Get the predicted class index
    predicted_class = torch.argmax(logits, dim=1).item()

    # Retrieve label name from mapping
    predicted_label = label_map.get(predicted_class, "Unknown")
    return predicted_label

# Streamlit app
def main():
    st.title("Flower Classification Prediction")
    st.write("Upload an image of a flower to predict its type.")

    # Image input
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Open and display the uploaded image
        img = Image.open(uploaded_image).convert("RGB")
        st.image(img, caption='Uploaded Image.', use_container_width=True)

        # Prediction button
        if st.button("Predict"):
            # Predict the label for the uploaded image
            predicted_label = predict_label(img)
            st.success(f"The predicted flower is: {predicted_label}")

if __name__ == "__main__":
    main()
