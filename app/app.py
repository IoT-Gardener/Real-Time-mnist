import logging
import requests
import pandas as pd
from PIL import Image
import streamlit as st
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

# Load assets
lottie_cv = load_assets("https://assets10.lottiefiles.com/private_files/lf30_dmituz7c.json")

# Set the page title and icon and set layout to "wide" to minimise margains
st.set_page_config(page_title="Real Time CV", page_icon=":1234:")

# Title and intro container
with st.container():
    # Add two columns for page formatting, with a width ratio of 1:1.75
    title_left_col, title_right_col = st.columns((2, 1))

    with title_left_col:
        st.title("Real Time Number Classification")
        st.write("Model score card and real time endpoint for a CNN trained to classify hand written numbers.")
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
    # Add an accuracy score card, with a tooltip
    col1.metric("Accuracy", "99%", help="the percentage of **correctly** classified samples of the **total** number of classified samples")
    # Add a precision scorecard, with a tooltip
    col2.metric("Avg. Precision", "99%", help="The percentage of **correctly** classified samples out of the classifications made for that class, Averaged across all classes")
    # Add a recall scorecard with a tooltip
    col3.metric("Avg. Recall", "99%", help="The percentage **correctly** classified samples out of the total samples for that class. Averaged across all classes.")
    # Add an F-1 score scorecard with a tooltip
    col4.metric("Avg. F-1 Score", "99", help="A performance metric calculated as the harmonic mean of precision and recall for a class. Averaged across all classes")
    st.write("---")

# Canvas container
with st.container():
    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=10,
        stroke_color="#000000",
        background_color="#ffffff",
        update_streamlit=True,
        height=250,
        width=250,
        drawing_mode="freedraw",
        point_display_radius=0,
        key="canvas"
    )

    # Do something interesting with the image data and paths
    # if canvas_result.image_data is not None:
    #     st.image(canvas_result.image_data)

# Footer section
with st.container():
    st.write("""
    **Author**: Alex.B, :wave: [LinkedIn](https://www.linkedin.com/in/alexander-billington-29488b118/) 
    :books: [Github](https://github.com/ABillington96) :computer: [Website](https://abmlops.streamlit.app)
    """)