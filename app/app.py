import json
import logging
import requests
import numpy as np
import pandas as pd
import streamlit as st
from keras.models import load_model
from pathlib import Path
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from streamlit_lottie import st_lottie


def load_assets(url: str) -> dict:
    """Function load asset from a http get request
    Args:
        url (str): URL of the asset to load
    Returns:
        dict: _description_
    """
    asset = requests.get(url)
    if asset.status_code != 200:
        logging.error("Failed to load asset")
        return None
        
    else:
        logging.info("Asset loaded successfully")
        return asset.json()
    

def format_image(img: np.ndarray) -> np.ndarray:
    """Function to convert an RBG image to grey scale

    Args:
        img (np.ndarray): The RGB input image to be converted

    Returns:
        np.ndarray: The output image in greyscale
    """
    # Convert the image into a PIL object
    pil_img = Image.fromarray(img)
    # Resize the image
    pil_img_r = pil_img.resize((28,28))
    # Convert the resized image back to an array
    rgb = np.array(pil_img_r)
    # Extract the red, green and blue channels from the image
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    # Convert into a single greyscale channel
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    # Normalise the image
    norm = gray / 255

    reshaped = norm.reshape(1, 28, 28, 1)
    return reshaped

# Load assets
lottie_cv = load_assets("https://assets10.lottiefiles.com/private_files/lf30_dmituz7c.json")
scorecard = data = json.load(open(f"{Path(__file__).parents[0]}/data/scorecard.json"))
model = load_model(f"{Path(__file__).parents[0]}/../app/data/final_model.h5")

# Set the page title and icon and set layout to "wide" to minimise margains
st.set_page_config(page_title="Real Time CV", page_icon=":1234:")

# Title and intro container
with st.container():
    # Add two columns for page formatting, with a width ratio of 1:1.75
    title_left_col, title_right_col = st.columns((2, 1))

    with title_left_col:
        st.title("Real Time Number Classification")
        st.write("Model scorecard and real time endpoint for a CNN trained to classify hand written numbers.")
    with title_right_col:
        # Add a lottie animation
        st_lottie(lottie_cv, key="cv animation", width=200)

    st.write("---")
    
# Scorecard container
with st.container():
    # Add a title
    st.header("Model Scorecard")
    st.write("A breakdown of the performance of the production model.")
    # Create 4 columns, one for each metric
    col1, col2, col3, col4 = st.columns(4)
    # Add an accuracy scordcard, with a tooltip
    col1.metric("Accuracy", f"{scorecard['accuracy']}%", 
                help="the percentage of **correctly** classified samples of the **total** number of classified samples")
    # Add a precision scorecard, with a tooltip
    col2.metric("Avg. Precision", f"{scorecard['precision']}%", 
                help="The percentage of **correctly** classified samples out of the classifications made for that class, Averaged across all classes")
    # Add a recall scorecard with a tooltip
    col3.metric("Avg. Recall", f"{scorecard['recall']}%", 
                help="The percentage **correctly** classified samples out of the total samples for that class. Averaged across all classes.")
    # Add an F-1 score scorecard with a tooltip
    col4.metric("Avg. F-1 Score", f"{scorecard['f1']}", 
                help="A performance metric calculated as the harmonic mean of precision and recall for a class. Averaged across all classes")
    st.write("---")

# Canvas container
with st.container():
    st.header("Real Time Classification")
    st.write("Please draw a number between 0 & 9 in the box below to recieve a classification.")
    model_left, model_right = st.columns((1,2))
    
    with model_left:
    # Create a canvas component
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
            stroke_width=10,
            stroke_color="#ffffff",
            background_color="#000000",
            update_streamlit=True,
            height=224,
            width=224,
            drawing_mode="freedraw",
            point_display_radius=0,
            key="canvas"
        )

    with model_right:
        # Do something interesting with the image data and paths
        if canvas_result.image_data is not None:

            # Get the output from the canvas
            model_input = format_image(canvas_result.image_data)
            # Query the model
            prediction = model.predict(model_input)
            
            df= pd.DataFrame({
                "Class": ["0","1", "2", "3", "4", "5", "6", "7", "8", "9"],
                "Probability": prediction[0]
            }   
            )
            st.bar_chart(prediction[0])

            

# Footer section
with st.container():
    st.write("""
    **Author**: Alex.B, :wave: [LinkedIn](https://www.linkedin.com/in/alexander-billington-29488b118/) 
    :books: [Github](https://github.com/ABillington96) :computer: [Website](https://abmlops.streamlit.app)
    """)
