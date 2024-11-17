import streamlit as st
from PIL import Image
import torch
import numpy as np
from torchvision.transforms.functional import to_pil_image
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Dictionary of model paths
MODEL_PATHS = {
    "Cabel Defect": "weights/cable_model.h5",
    "Hazelnut Defect": "weights/hazelnut_model.h5",
    "Leather Defect": "weights/leather_model.h5",
    "Pill Defect": "weights/pill_model.h5",
    "Toothbrush Defect": "weights/toothbrush_model.h5"
}

@st.cache_resource
def load_all_models():
    """Load all models and cache them"""
    models = {}
    for model_name, model_path in MODEL_PATHS.items():
        try:
            model = torch.load(model_path, map_location=torch.device("cpu"))
            model.eval()
            models[model_name] = model
        except Exception as e:
            st.error(f"Error loading {model_name}: {str(e)}")
    return models

def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return transform(image)

def get_bbox_from_heatmap(heatmap, thres=0.8):
    # Ensure heatmap is 2D
    if len(heatmap.shape) > 2:
        heatmap = np.mean(heatmap, axis=0)
    
    # Create binary map
    binary_map = heatmap > thres
    
    # Find non-zero coordinates
    y_coords, x_coords = np.nonzero(binary_map)
    
    if len(x_coords) == 0 or len(y_coords) == 0:
        return 0, 0, binary_map.shape[1], binary_map.shape[0]
    
    # Get bounding box coordinates
    x_0 = np.min(x_coords)
    x_1 = np.max(x_coords)
    y_0 = np.min(y_coords)
    y_1 = np.max(y_coords)
    
    # Add padding
    padding = 5
    x_0 = max(0, x_0 - padding)
    y_0 = max(0, y_0 - padding)
    x_1 = min(binary_map.shape[1], x_1 + padding)
    y_1 = min(binary_map.shape[0], y_1 + padding)
    
    return x_0, y_0, x_1, y_1

def predict_and_localize(model, image, device, threshold=0.8):
    # Transform input image
    input_tensor = transform_image(image).unsqueeze(0).to(device)
    
    # Perform prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        prediction_probs, heatmap_tensor = outputs[0], outputs[1]
    
    # Process results
    class_idx = torch.argmax(prediction_probs, dim=-1).item()
    probability = torch.max(prediction_probs).item()
    
    # Convert heatmap to numpy and ensure correct shape
    heatmap = heatmap_tensor.squeeze().cpu().numpy()
    if len(heatmap.shape) == 3:
        heatmap = np.mean(heatmap, axis=0)
    
    # Get bounding box if anomaly detected
    bounding_box = get_bbox_from_heatmap(heatmap, threshold) if class_idx == 1 else None
    
    return class_idx, probability, heatmap, bounding_box

def display_results(image, class_idx, probability, heatmap, bbox, model_name):
    """Display prediction results for a specific model"""
    st.write(f"### Results for {model_name}")
    
    # Display prediction results
    labels = ["Normal", "Defective"]
    st.write(f"Prediction: {labels[class_idx]}")
    st.write(f"Probability: {probability:.3f}")
    
    # Create figure for visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(np.array(image))
    heatmap_display = ax.imshow(heatmap, cmap="Reds", alpha=0.5)
    
    if bbox:
        x_0, y_0, x_1, y_1 = bbox
        rect = Rectangle((x_0, y_0), x_1 - x_0, y_1 - y_0,
                       edgecolor="red", facecolor="none", lw=2)
        ax.add_patch(rect)
        st.write(f"Defect Bounding Box: {bbox}")
    
    ax.axis("off")
    st.pyplot(fig)
    plt.close()

def main():
    st.title("Multi-Model Defect Detection and Localization")
    st.write("Upload an image and select a model for prediction.")
    
    # Load all models
    models = load_all_models()
    
    # File uploader
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        # Display uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Create columns for model buttons
        cols = st.columns(len(models))
        
        # Device selection
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create a button for each model
        for col, (model_name, model) in zip(cols, models.items()):
            with col:
                if st.button(f"Predict with {model_name}"):
                    with st.spinner(f'Running prediction with {model_name}...'):
                        try:
                            # Perform prediction
                            class_idx, probability, heatmap, bbox = predict_and_localize(
                                model, image, device
                            )
                            
                            # Display results
                            display_results(
                                image, class_idx, probability, heatmap, bbox, model_name
                            )
                            
                        except Exception as e:
                            st.error(f"Error during prediction with {model_name}: {str(e)}")

if __name__ == "__main__":
    main()