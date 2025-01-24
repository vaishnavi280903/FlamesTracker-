import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from streamlit_option_menu import option_menu  # Install this package using 'pip install streamlit-option-menu'

# Load the trained model
@st.cache_resource
def load_trained_model():
    model = load_model('wildfire.h5')  # Ensure this path is correct
    return model

model = load_trained_model()

# Image preprocessing function
IMG_SIZE = (128, 128)  # Same size used during training

def preprocess_image(image):
    try:
        img = image.resize(IMG_SIZE)  # Resize image
        img_array = img_to_array(img)  # Convert image to array
        img_array = img_array / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array
    except Exception as e:
        st.error(f"Error during image preprocessing: {e}")
        return None

# Sidebar Menu Bar
st.sidebar.title("Menu Bar")
menu_selection = st.sidebar.radio(
    "Choose an Option:",
    ["Upload Image", "Prediction History", "System Info"]
)

# Upload Image functionality
if menu_selection == "Upload Image":
    st.sidebar.info("Upload an image for wildfire detection.")
    st.title("Upload Image for Wildfire Detection")

    # Upload the image
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Predict button
        if st.button("Predict"):
            with st.spinner("Processing Image..."):
                try:
                    # Preprocess and predict
                    image = load_img(uploaded_file)
                    processed_image = preprocess_image(image)
                    if processed_image is not None:
                        prediction = model.predict(processed_image)[0][0]  # Extract single prediction value

                        # Display prediction
                        threshold = 0.5  # Adjust the threshold as needed
                        if prediction > threshold:
                            st.error(f"🔥**Wildfire Detected**🔥")
                            st.warning("Take action immediately to prevent spread!")
                            # Save prediction to session state
                            if "history" not in st.session_state:
                                st.session_state["history"] = []
                            st.session_state["history"].append(("Wildfire Detected", uploaded_file.name))
                        else:
                            st.success("✅ No Wildfire Detected")
                            st.balloons()
                            if "history" not in st.session_state:
                                st.session_state["history"] = []
                            st.session_state["history"].append(("No Wildfire Detected", uploaded_file.name))
                except Exception as e:
                    st.error(f"Error during prediction: {e}")

# Prediction History functionality
elif menu_selection == "Prediction History":
    st.sidebar.info("Check your recent predictions.")
    st.title("Prediction History")

    # Check if history exists
    if "history" in st.session_state and st.session_state["history"]:
        st.write("Here are your recent predictions:")
        for i, (result, filename) in enumerate(reversed(st.session_state["history"]), start=1):
            st.write(f"{i}. **Image:** {filename}, **Result:** {result}")
    else:
        st.warning("No predictions available. Please upload an image first.")

# System Info functionality
elif menu_selection == "System Info":
    st.sidebar.info("View details about this application.")
    st.title("System Information")

    # Display app and system details
    st.write("### Application Details")
    st.write("- **Name**: Wildfire Detection System")
    st.write("- **Version**: 1.0")
    st.write("- **Description**: Detects wildfires in satellite images using a trained deep learning model.")
    
    st.write("### System Information")
    st.write(f"- **Python Version**: {st.__version__}")
    st.write(f"- **TensorFlow Version**: 2.9.0 or above (ensure compatibility)")
    st.write(f"- **Platform**: Streamlit Cloud or Local Deployment")

    # Add a hardware/software info image
    st.image(
        "https://cdn-icons-png.flaticon.com/512/2920/2920076.png",
        caption="System Details",
        use_column_width=True,
    )

# Footer section
st.markdown(
    """
    <div style="text-align: center; margin-top: 50px; font-size: 14px; color: gray;">
        Built with ❤️ by Vaishnavi. Stay safe and vigilant!
    </div>
    """,
    unsafe_allow_html=True,
)
