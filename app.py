import streamlit as st
import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
import os

from google.colab import drive
drive.mount('/content/drive')

model_path = 'https://drive.google.com/file/d/1fftXlxFl-LPcA3SGENUwwVxyQu1uNWNK/view?usp=drive_link'
model = load_model(model_path)
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)
class_labels = {v: k for k, v in class_indices.items()}  # Invert the class indices

# Add Groq initialization
groq_api_key = os.getenv('GROQ_API_KEY')  # Make sure to set this environment variable
chat_model = ChatGroq(api_key=groq_api_key)

# Function to preprocess the image
def preprocess_image(image):
    img = image.resize((224, 224))                       # Resize image to match model input
    img_array = img_to_array(img) / 255.0                # Convert to array and normalize
    img_array = np.expand_dims(img_array, axis=0)        # Expand dimensions to match model input shape
    return img_array

# Function to make a prediction
def predict_image_class(img_array):
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    return class_labels[predicted_class]

# Add function to get disease information
def get_disease_info(disease_name):
    # Check if the prediction indicates a healthy plant
    if 'healthy' in disease_name.lower():
        return """
        This plant appears to be healthy! 
        
        General maintenance tips:
        1. Continue regular watering and proper nutrition
        2. Maintain good air circulation
        3. Monitor for any changes in leaf color or texture
        4. Follow regular pruning practices
        """
    
    # For disease cases, continue with the original prompt
    prompt = f"""
    Provide information about the plant disease '{disease_name}' in the following format:
    1. Brief description in simple language (2-3 sentences)
    2. Key prevention measures (3-4 points)
    Keep the response concise and easy to understand.
    """
    
    messages = [HumanMessage(content=prompt)]
    response = chat_model.invoke(messages)
    return response.content

# Streamlit interface
st.title("Plant Disease Classification")
st.write("Upload an image of a fruit or vegetable leaf, and the model will predict the disease.")

# Create sidebar for file upload
with st.sidebar:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Display uploaded image and prediction
if uploaded_file is not None:
    # Create two columns - one for image, one for prediction
    col1, col2 = st.columns([1, 1])  # Equal width columns
    
    # Open and resize image
    image = Image.open(uploaded_file)
    # Resize image to a smaller size while maintaining aspect ratio
    max_size = (400, 400)
    image.thumbnail(max_size, Image.LANCZOS)
    
    # Display image and button in left column
    with col1:
        st.image(image, caption="Uploaded Image")
        predict_button = st.button("Predict")
    
    # Prediction content in right column
    with col2:
        if predict_button:  # Changed from if st.button("Predict")
            # Preprocess and predict
            img_array = preprocess_image(image)
            prediction = predict_image_class(img_array)
            
            # Display prediction first
            st.write(f"### Prediction:")
            st.write(f"{prediction}")
            
            # Get and display disease information using Langchain + Groq
            with st.spinner("Getting disease information..."):
                disease_info = get_disease_info(prediction)
                st.write("### Disease Information:")
                st.write(disease_info)
