import json
import keras
import logging
import mlflow
import requests
import numpy as np
import pandas as pd
import streamlit as st
from databricks_cli.sdk.api_client import ApiClient
from mlflow.tracking import MlflowClient
from pathlib import Path
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from streamlit_lottie import st_lottie

# Connect to the databricks API client
api_client = ApiClient(
  host  = st.secrets["DATABRICKS_HOST"],
  token = st.secrets["DATABRICKS_TOKEN"]
)

# Set the mlflow tracking to databricks managed mlflow
mlflow_uri = "databricks"
client = MlflowClient(tracking_uri=mlflow_uri)
mlflow.set_tracking_uri(mlflow_uri)


@st.cache_resource
def load_model() -> keras.engine.sequential.Sequential:
    """
    Function to load the model from MLFlow and assign it to the global MODEL variable
    Returns:
        keras.engine.sequential.Sequential: mnist model
    """
    # Load the model from MLFlow
    try:
        model = mlflow.tensorflow.load_model(model_uri="models:/mnist-tf-cnn/Production")
        logging.info("Successfully loaded model")
    # If unable to load model, print error and set model to None
    except Exception as e:
        logging.error(f"Error loading model, {e}")
        model = None

    return model


@st.cache_resource
def get_model_metrics() -> dict:
    try:
        # Iterate over all register models
        for mv in client.search_model_versions("name='mnist-tf-cnn'"):
            # Store the run id of the production model
            if dict(mv)["current_stage"] == "Production":
                run_id = dict(mv)['run_id']
        # Create a dictionary to store the scorecard
        scorecard = {}
        # Get the value of the metric from the most recent time step for each metric
        scorecard["accuracy"] = dict(client.get_metric_history(run_id, "accuracy")[-1])['value']
        scorecard["precision"] = dict(client.get_metric_history(run_id, "precision")[-1])['value']
        scorecard["recall"] = dict(client.get_metric_history(run_id, "recall")[-1])['value']
        scorecard["f1"] = dict(client.get_metric_history(run_id, "f1")[-1])['value']
        logging.info("Loaded model scorecard")
        return scorecard
    except Exception as e:
        logging.error(f"Failed to load model metrics, {e}")
        return None


@st.cache_resource
def load_assets(url: str) -> dict:
    """Function load asset from a http get request
    Args:
        url (str): URL of the asset to load
    Returns:
        dict: Lottie media json
    """
    # Request the asset from lottie
    asset = requests.get(url)
    # If the request is not successful, return an error
    if asset.status_code != 200:
        logging.error("Failed to load asset")
        return {"Response": "Error"}
    # Otherwise return the json format of the media
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

# Set the page title and icon and set layout to "wide" to minimise margains
st.set_page_config(page_title="Real Time CV", page_icon=":1234:")

# Load assets
lottie_cv = load_assets("https://assets10.lottiefiles.com/private_files/lf30_dmituz7c.json")
model = load_model()
scorecard = get_model_metrics()

# Title and intro container
with st.container():
    # Add two columns for page formatting, with a width ratio of 1:1.75
    title_left_col, title_right_col = st.columns((2, 1))

    with title_left_col:
        st.title("Real Time Number Classification")
        st.write("Model scorecard and real time endpoint for a CNN trained to classify hand written numbers.")
    with title_right_col:
        if "Response" in lottie_cv.keys() and lottie_cv["Response"] == "Error":
            pass
        else:
            # Add a lottie animation
            st_lottie(lottie_cv, key="cv animation", width=200)

    st.write("---")

# Check if a model was loaded
if model==None:
    # If no model was loaded inform the user
    with st.container():
        st.write("Unable to load production model, please try again later.")
        # Create button to try loading the model again
        if st.button("Try again"):
            st.cache_resource.clear()

else:
    # Scorecard container
    with st.container():
        # Add a title
        st.header("Model Scorecard")
        st.write("A breakdown of the performance of the production model.")
        if scorecard == None:
            st.write("No scorecard data could be loaded, please reload try again later using the 'reload data' button below")
        else:
            # Create 4 columns, one for each metric
            col1, col2, col3, col4 = st.columns(4)
            # Add an accuracy scordcard, with a tooltip
            col1.metric("Accuracy", f"{round(scorecard['accuracy'],4)}", 
                        help="the percentage of **correctly** classified samples of the **total** number of classified samples")
            # Add a precision scorecard, with a tooltip
            col2.metric("Avg. Precision", f"{round(scorecard['precision'],4)}", 
                        help="The percentage of **correctly** classified samples out of the classifications made for that class, Averaged across all classes")
            # Add a recall scorecard with a tooltip
            col3.metric("Avg. Recall", f"{round(scorecard['recall'],4)}", 
                        help="The percentage **correctly** classified samples out of the total samples for that class. Averaged across all classes.")
            # Add an F-1 score scorecard with a tooltip
            col4.metric("Avg. F-1 Score", f"{round(scorecard['f1'],4)}", 
                        help="A performance metric calculated as the harmonic mean of precision and recall for a class. Averaged across all classes")
        st.write("---")

    # Canvas container
    with st.container():
        # Add section headers and description
        st.header("Real Time Classification")
        st.write("Please draw a number between 0 & 9 in the box below to recieve a classification.")

        # Create two columns
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
            if st.button("Reload data"):
                st.cache_resource.clear()

        with model_right:
            # Do something interesting with the image data
            if canvas_result.image_data is not None:
                # Get the output from the canvas
                model_input = format_image(canvas_result.image_data)
                # Query the model
                prediction = model.predict(model_input)
                # Display a barchart of the predictions
                st.bar_chart(prediction[0])

# Footer section
with st.container():
    st.write("""
    **Author**: Alex.B, :wave: [LinkedIn](https://www.linkedin.com/in/alexander-billington-29488b118/) 
    :books: [Github](https://github.com/IoT-Gardener) :computer: [Website](https://abmlops.streamlit.app)
    """)
