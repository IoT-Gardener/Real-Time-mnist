# Real-Time-mnist
Using Streamlit &amp; MLFlow to serve real time computer vision models.

# Setup
Below are a set of instruction on how to set up the various tools and services required.

## Python virtual environment
All of the following commands should be run in a terminal on a machine with python installed a python download can be found [here](https://www.python.org/downloads/).
1) Create the virtual environment:
```
py -m venv .venv
```
2) Activate the virtual enviornment:
```
.\.venv\Scripts\activate
```
3) Done. It is as easy as that!
Bonus step is to install of all the required python packages from the requirements.txt
- Install the requirements:
```
pip install -r requirements.txt
```

## Setup Databricks MLFlow
This implementation uses managed MLFlow, part of Databricks and it is very easy to configure. If you do not already have databricks you can get the community version [here](https://www.databricks.com/try-databricks#account).

First you will need to do three things:
1) Get your Databricks host URL
2) Generate an Access token, by going to *User Settings* -> *Acess tokens* -> *Generate new token*
3) Create an experiment call MNIST, by going to *Experiment* -> *Create Blank Experiment*, and get the location

Once you ahve done this there are two small setup steps left.

### Configure Local Environment for Model Training
1) Run the databricks configuration
```
databricks configure --token
```
2) Enter your host and token when prompted
3) Create an environment variable for your experiment location
```
setx MNIST_EXPERIMENT="<experiment_url>"
```

### Configure Streamlit
To allow the Streamlit app to access MLFlow it logs into the databricks api_client using credentials stored in a streamlit secrets file.
1) Create a file called *secrets.toml* in the *.streamlit* folder
2) Add two lines to the .toml file:
```
DATABRICKS_HOST="<host_url>"
DATABRICKS_TOKEN="<token>"
```

## Run the Streamlit app
1) Navigate in the src folder using:
```
cd app
```
2) Run the app
```
streamlit run app.py