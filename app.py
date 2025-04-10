import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import os
import cv2
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="PCOS Detection System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .header {
        color: #1E3A8A;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-align: center;
    }
    .subheader {
        color: #475569;
        font-size: 1.2rem;
        font-weight: 400;
        margin-bottom: 2rem;
        text-align: center;
    }
    .card {
        background-color: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .result-positive {
        background-color: #FEE2E2;
        color: #B91C1C;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
    }
    .result-negative {
        background-color: #DCFCE7;
        color: #166534;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
    }
    .warning {
        color: #EA580C;
        font-size: 0.85rem;
        font-style: italic;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        font-size: 0.8rem;
        color: #6B7280;
    }
</style>
""", unsafe_allow_html=True)

# Constants
IMG_SIZE_VGG_RESNET = (224, 224)  # VGG16 and ResNet50 use 224x224
IMG_SIZE_INCEPTION = (299, 299)    # InceptionV3 uses 299x299

#path to your models
def get_model_paths():
    return {
        'vgg16': 'C:\\Users\\Jasmitha\\Downloads\\PCOS (1)\\best_model_vgg16.keras',
        'resnet50': 'C:\\Users\\Jasmitha\\Downloads\\PCOS (1)\\best_model_resnet50.keras',
        'inception': 'C:\\Users\\Jasmitha\\Downloads\\PCOS (1)\\best_model_inception.keras',
        'custom': 'C:\\Users\\Jasmitha\\Downloads\\PCOS (1)\\best_model.keras'
    }



OPTIMAL_WEIGHTS = np.array([0.3, 0.3, 0.2, 0.2]) 

@st.cache_resource
def load_models():
    """Load all the models and cache them"""
    models = {}
    model_paths = get_model_paths()
    
    with st.spinner("Loading models (this may take a minute)..."):
        try:
            models['vgg16'] = load_model(model_paths['vgg16'])
            st.success("✅ VGG16 model loaded")
        except Exception as e:
            st.error(f"❌ Error loading VGG16 model: {e}")
            
        try:
            models['resnet50'] = load_model(model_paths['resnet50'])
            st.success("✅ ResNet50 model loaded")
        except Exception as e:
            st.error(f"❌ Error loading ResNet50 model: {e}")
            
        try:
            models['inception'] = load_model(model_paths['inception'])
            st.success("✅ InceptionV3 model loaded")
        except Exception as e:
            st.error(f"❌ Error loading InceptionV3 model: {e}")
            
        try:
            models['custom'] = load_model(model_paths['custom'])
            st.success("✅ Custom CNN model loaded")
        except Exception as e:
            st.error(f"❌ Error loading Custom CNN model: {e}")
    
    return models

# Function to preprocess an image for a specific model
def preprocess_image(img, target_size):
    """Preprocess an image for model prediction"""
    # Resize image
    img = img.resize(target_size)
    
    # Convert to array and normalize
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    return img_array

# Function to make prediction with the ensemble model
def predict_ensemble(img, models):
    """Make a prediction with the ensemble model"""
    predictions = []
    model_names = []
    
    # VGG16 prediction
    if 'vgg16' in models:
        img_vgg = preprocess_image(img, IMG_SIZE_VGG_RESNET)
        pred_vgg = models['vgg16'].predict(img_vgg)[0][0]
        predictions.append(pred_vgg)
        model_names.append('VGG16')
    
    # ResNet50 prediction
    if 'resnet50' in models:
        img_resnet = preprocess_image(img, IMG_SIZE_VGG_RESNET)
        pred_resnet = models['resnet50'].predict(img_resnet)[0][0]
        predictions.append(pred_resnet)
        model_names.append('ResNet50')
    
    # InceptionV3 prediction
    if 'inception' in models:
        img_inception = preprocess_image(img, IMG_SIZE_INCEPTION)
        pred_inception = models['inception'].predict(img_inception)[0][0]
        predictions.append(pred_inception)
        model_names.append('InceptionV3')
    
    # Custom CNN prediction
    if 'custom' in models:
        img_custom = preprocess_image(img, IMG_SIZE_VGG_RESNET)
        pred_custom = models['custom'].predict(img_custom)
        
        # Handle categorical output if needed
        if len(pred_custom.shape) > 1 and pred_custom.shape[1] > 1:
            pred_custom = pred_custom[0][1]  # Get probability of positive class
        else:
            pred_custom = pred_custom[0][0]
            
        predictions.append(pred_custom)
        model_names.append('Custom CNN')
    
    # If we have fewer models than expected weights, adjust weights
    weights = OPTIMAL_WEIGHTS[:len(predictions)]
    if sum(weights) != 1.0:
        weights = weights / sum(weights)
    
    # Calculate weighted ensemble prediction
    if predictions:
        ensemble_pred = sum(p * w for p, w in zip(predictions, weights))
        predictions.append(ensemble_pred)
        model_names.append('Ensemble')
    else:
        st.error("No models available for prediction")
        return None, None
    
    # Create a DataFrame for displaying model predictions
    # UPDATED: Flipped interpretation - 0 is PCOS (infected), 1 is Normal (not infected)
    results_df = pd.DataFrame({
        'Model': model_names,
        'Probability': predictions,
        'Prediction': ['Normal' if p >= 0.5 else 'PCOS' for p in predictions]
    })
    
    return ensemble_pred, results_df

# Function to create prediction visualization
def create_prediction_chart(results_df):
    """Create a bar chart for model predictions"""
    # UPDATED: Color coding - higher probability now means Normal (not infected)
    results_df['Color'] = results_df['Probability'].apply(
        lambda x: '#166534' if x >= 0.5 else '#B91C1C'
    )
    
    # Create the bar chart
    fig = px.bar(
        results_df, 
        x='Model', 
        y='Probability',
        text='Probability',
        color='Color',
        color_discrete_map="identity",  # Use the actual color values
        title="Model Predictions",
        height=400
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title="Model",
        yaxis_title="Probability (Higher = Normal, Lower = PCOS)",  # Updated label
        yaxis=dict(range=[0, 1]),
        plot_bgcolor='white',
        font=dict(family="Arial, sans-serif", size=14),
        showlegend=False
    )
    
    # Format the text labels
    fig.update_traces(
        texttemplate='%{y:.2f}',
        textposition='outside'
    )
    
    # Add a threshold line at 0.5
    fig.add_shape(
        type="line",
        x0=-0.5,
        x1=len(results_df)-0.5,
        y0=0.5,
        y1=0.5,
        line=dict(color="black", width=2, dash="dash")
    )
    
    # Add annotation for the threshold
    fig.add_annotation(
        x=len(results_df)-1,
        y=0.52,
        xref="x",
        yref="y",
        text="Threshold (0.5)",
        showarrow=False,
        font=dict(size=12)
    )
    
    return fig

# Function to create a gauge chart for the ensemble prediction
def create_gauge_chart(ensemble_pred):
    """Create a gauge chart for the ensemble prediction"""
    # UPDATED: Colors and interpretation for gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=ensemble_pred,
        title={'text': "Probability (Higher = Normal, Lower = PCOS)"},
        gauge={
            'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                # UPDATED: Color scheme is now reversed
                {'range': [0, 0.3], 'color': 'red'},     # Low values = PCOS (red)
                {'range': [0.3, 0.7], 'color': 'yellow'},
                {'range': [0.7, 1], 'color': 'green'}    # High values = Normal (green)
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.5
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        font=dict(family="Arial, sans-serif", size=14)
    )
    
    return fig

# Main app function
def main():
    # Header
    st.markdown('<div class="header">PCOS Detection System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subheader">Hybrid deep learning system for Polycystic Ovary Syndrome (PCOS) detection from ultrasound images</div>', 
        unsafe_allow_html=True
    )
    
    # Sidebar
    with st.sidebar:
        st.markdown("### About PCOS")
        st.info(
            "Polycystic Ovary Syndrome (PCOS) is a hormonal disorder common among women of reproductive age. "
            "Early detection through ultrasound imaging can help in better management of the condition."
        )
        
        st.markdown("### Model Information")
        st.write("This application uses an ensemble of four deep learning models:")
        st.write("- VGG16 (transfer learning)")
        st.write("- ResNet50 (transfer learning)")
        st.write("- InceptionV3 (transfer learning)")
        st.write("- Custom CNN")
        
        # ADDED: Model output interpretation explanation
        st.markdown("### Model Output Interpretation")
        st.info(
            "In this model:\n"
            "- Values close to 0 indicate PCOS (infected)\n"
            "- Values close to 1 indicate Normal (not infected)"
        )
        
        st.markdown("### Disclaimer")
        st.warning(
            "This tool is for educational purposes only and should not replace professional medical advice. "
            "Always consult with a healthcare provider for diagnosis and treatment."
        )
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Upload Ultrasound Image")
        st.write("Upload an ultrasound image for PCOS detection:")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Display the uploaded image
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded Image")
            
            # Add a predict button
            predict_button = st.button("Analyze Image")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Load models once
    models = load_models()
    
    # Make prediction if button is clicked
    if 'predict_button' in locals() and predict_button and uploaded_file is not None:
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Analysis Results")
            
            # Show a spinner while predicting
            with st.spinner("Analyzing image..."):
                # Simulate processing time
                progress_bar = st.progress(0)
                for i in range(101):
                    time.sleep(0.01)
                    progress_bar.progress(i)
                
                # Make prediction
                ensemble_pred, results_df = predict_ensemble(img, models)
            
            if ensemble_pred is not None:
                # UPDATED: Show the ensemble prediction with flipped interpretation
                if ensemble_pred >= 0.5:
                    st.markdown(
                        f'<div class="result-negative">Normal with {ensemble_pred:.1%} probability</div>', 
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="result-positive">PCOS Detected with {1-ensemble_pred:.1%} probability</div>', 
                        unsafe_allow_html=True
                    )
                
                # Show the gauge chart
                st.plotly_chart(create_gauge_chart(ensemble_pred), use_container_width=True)
                
                # Show the prediction results
                st.subheader("Model Predictions")
                st.dataframe(results_df[['Model', 'Probability', 'Prediction']], use_container_width=True)
                
                # Show the prediction chart
                st.plotly_chart(create_prediction_chart(results_df), use_container_width=True)
                
                # Add a warning/disclaimer
                st.markdown(
                    '<div class="warning">Note: This analysis is for informational purposes only. '
                    'Please consult with a healthcare professional for diagnosis.</div>', 
                    unsafe_allow_html=True
                )
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Additional information section
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("About PCOS Detection")
    st.write(
        "Polycystic Ovary Syndrome (PCOS) affects approximately 1 in 10 women of reproductive age worldwide. "
        "It is characterized by hormonal imbalances and metabolism problems that may affect a woman's overall health. "
        "Key features visible in ultrasound include multiple small follicles in the ovaries."
    )
    
    # Two columns for symptoms and risk factors
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        st.markdown("#### Common Symptoms")
        st.markdown("- Irregular periods")
        st.markdown("- Excess androgen levels")
        st.markdown("- Polycystic ovaries")
        st.markdown("- Weight gain")
        st.markdown("- Acne or oily skin")
        st.markdown("- Hair thinning or hair loss")
    
    with info_col2:
        st.markdown("#### Risk Factors")
        st.markdown("- Family history of PCOS")
        st.markdown("- Insulin resistance")
        st.markdown("- Obesity")
        st.markdown("- Inflammation")
        st.markdown("- Exposure to certain medications")
    
    st.write(
        "Early detection and management of PCOS is important for preventing long-term complications "
        "including type 2 diabetes, heart disease, hypertension, and endometrial cancer."
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown(
        '<div class="footer">© 2025 PCOS Detection System | This application is for educational purposes only.</div>', 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()