## Requirements ##

This app requires the following Python packages:

- Flask : for building the web framework
- pandas : for data analysis and manipulation
- matplotlib : for generating plots and visualizations
- scikit-learn : for statistical modeling and machine learning
- flask-cors : for handling cross-origin requests from the frontend

---------------------------------------------------------------------------
## Installation ##

- Clone the repository:
git clone https://github.com/etc

- Navigate into the project directory and create a virtual environment
cd filepath
python -m venv venv

- Activate the virtual environment
# Windows
venv\Scripts\activate
# Linux/MacOS
source venv/bin/activate

- Install the required packages:
pip install -r requirements.txt

------------------------------------------------------------------------------
## Running the App ##

- With the virtual environment activated, start the Flask app:
python app.py

The app will be available at http://localhost:5000
Upload a CSV file to explore and visualize the data

------------------------------------------------------------------------------
## Frontend ##

The frontend code lives in the static and templates directories
Modify and add JS/CSS/HTML as needed for the frontend