from flask import Flask, render_template, request, redirect, url_for, jsonify, session
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import os
import plotly
import plotly.graph_objs as go
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import json
import logging
from logging.handlers import RotatingFileHandler


app = Flask(__name__)

# Set the secret key for the session
app.secret_key = 'your_secret_key'

try:
    if not os.path.exists('logs'):
        os.makedirs('logs')
except Exception as e:
    print(f"An error occurred while trying to create the directory: {e}")

#----------------------------------------------------Logging
handler = RotatingFileHandler('logs\\app.log', maxBytes=10000, backupCount=10)
handler.setLevel(logging.INFO)
app.logger.addHandler(handler)

@app.route('/')
def index():
    return render_template('index.html')

#-------------------------------------------------------Load HTML tabs
@app.errorhandler(Exception)
def handle_exception(e):
    app.logger.error('%s', (e))
    return jsonify({'error': 'A server error occurred.', 'details': str(e)}), 500

@app.route('/datasummary')
def data_summary():
    # Render the data_summary.html template or generate the HTML content dynamically
    return render_template('data_summary.html')

@app.route('/dataanalysis')
def data_analysis():
    # Render the data_analysis.html template or generate the HTML content dynamically
    return render_template('data_analysis.html')

@app.route('/geoimaging')
def geo_imaging():
    # Render the geo_imaging.html template or generate the HTML content dynamically
    return render_template('geo_imaging.html')

@app.route('/machinelearning')
def machine_learning():
    # Render the machine_learning.html template or generate the HTML content dynamically
    return render_template('machine_learning.html')

@app.route('/trainedmodels')
def trained_models():
    # Render the trained_models.html template or generate the HTML content dynamically
    return render_template('trained_models.html')

#---------------------------------------------------------------Get DF Function
def get_data_and_columns(file):
    db_dir = './database'
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
    db_name = os.path.splitext(file.filename)[0] + '_database.db'
    table_name = os.path.splitext(file.filename)[0] + '_table'
    db_path = os.path.join(db_dir, db_name)
    engine = create_engine('sqlite:///' + db_path)
    df = pd.read_csv(file)
    df.to_sql(table_name, engine, if_exists='replace')
    session['db_name'] = db_name
    session['table_name'] = table_name
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df.select_dtypes(include=[object]).columns.tolist()
    session['numeric_columns'] = numeric_columns
    session['categorical_columns'] = categorical_columns

#---------------------------------------------------Display Functions
@app.route('/table_display', methods=['GET', 'POST'])
def table_display():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            get_data_and_columns(file)
            df = pd.read_sql_table(session['table_name'], create_engine('sqlite:///' + os.path.join('./database', session['db_name'])))
            return render_template('data_summary.html', tables=[df.to_html(classes='data')], titles=df.columns.values, numeric_columns=session['numeric_columns'], categorical_columns=session['categorical_columns'])
        else:
            return "File type is incorrect. Please upload a .csv file."
    else:
        # When a GET request is made, clear the session and render the template without any tables or columns
        session.clear()
        return render_template('data_summary.html')

#------------------------------------------------------------------Upload Function
@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file and allowed_file(file.filename):
        get_data_and_columns(file)
        db_path = os.path.join('./database', session['db_name'])
        engine = create_engine('sqlite:///' + db_path)
        df = pd.read_sql_table(session['table_name'], engine)
        return render_template('data_summary.html', tables=[df.to_html(classes='data')], numeric_columns=session['numeric_columns'], categorical_columns=session['categorical_columns'])
    else:
        return jsonify({'error': 'File type is incorrect. Please upload a .csv file.'}), 400
        
#-------------------------------------------------------------JSONify Columns Function
# Define the '/jsonify_columns' endpoint
@app.route('/jsonify_columns', methods=['GET'])
def jsonify_columns():
    db_dir = './database'
    db_name = session.get('db_name')  # Get database name from session data
    table_name = session.get('table_name')  # Get table name from session data
    if db_name is None or table_name is None:
        return jsonify(numeric_columns=[], categorical_columns=[])
    db_path = os.path.join(db_dir, db_name)
    engine = create_engine('sqlite:///' + db_path)
    df = pd.read_sql_table(table_name, engine)
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df.select_dtypes(include=[object]).columns.tolist()
    return jsonify(numeric_columns=numeric_columns, categorical_columns=categorical_columns)

#-----------------------------------------------------------------Plot Function (EDA)
@app.route('/plot', methods=['POST'])
def plot():
    print(request.get_json())
    if request.is_json:
        # Proceed if the request data is in JSON format
        data = request.get_json()
        # Check if 'column1' is in the request data
        if 'column1' not in data:
            return jsonify({"error": "Missing 'column1' in request data."}), 400
        column1 = data['column1']
        print(column1)
        column2 = data.get('column2')  # Use .get() to avoid KeyError if 'column2' is not in data
        plot_type = data['plot_type']
        db_dir = './database'
        db_name = session.get('db_name')  # Get database name from session data
        table_name = session.get('table_name')  # Get table name from session data
        db_path = os.path.join(db_dir, db_name)
        engine = create_engine('sqlite:///' + db_path)
        df = pd.read_sql_table(table_name, engine)
        # Profile and encode dataframe
        profile = profile_categorical_data(df)
        df_encoded, encoder_dict = smart_encode(df, profile)
        if column1 == 'ALL':
            columns = df_encoded.columns.tolist()
        else:
            columns = [column1, column2]
        if plot_type == 'histogram':
            data = [go.Histogram(x=df[columns[0]])]
        elif plot_type == 'box':
            data = [go.Box(y=df[columns[0]])]
        elif plot_type == 'bar':
            data = [go.Bar(x=df[columns[0]].value_counts().index, y=df[columns[1]].value_counts().values)]
        elif plot_type == 'heatmap':
            data = [go.Heatmap(z=df_encoded[columns].corr().values, x=columns, y=columns, colorscale='Viridis')]
        graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
        # Return the graphJSON data as a JSON response
        return jsonify({'graphJSON': graphJSON})
        
    else:
        # Return an error response if the request data is not in JSON format
        return jsonify({"error": "Invalid data format. Expected JSON."}), 400

#-------------------------------------------------------------Profile / Encode Data Function
def profile_categorical_data(df):
    """ 
    Profiles cat columns in panda dataframe to determine if they 
    should be one-hot encoded (nominal) or label encoded (ordinal) 
    """
    profile = {}
    for column in df.select_dtypes(include=['object', 'category', 'int']).columns:
        unique_values = df[column].dropna().unique()
        unique_values_count = len(unique_values)
        # Check if unique values have natural ordering
        if pd.api.types.is_integer_dtype(df[column]):
            # Assume ordinal if integers and range matches count of unique values
            if unique_values_count == df[column].max() - df[column].min() + 1:
                profile[column] = 'ordinal'
            else:
                profile[column] = 'nominal'
        elif pd.api.types.is_string_dtype(df[column]) or pd.api.types.is_categorical_dtype(df[column]):
            # Check if column can be converted to a numeric type (simple heuristic for ordinal)
            try:
                pd.to_numeric(df[column], errors='raise')
                profile[column] = 'ordinal'
            except ValueError:
                profile[column] = 'nominal'
        else:
            # Default to nominal if no clear ordinal pattern is detected
            profile[column] = 'nominal'
    return profile


def smart_encode(df, profile):
    le = LabelEncoder()
    """ Encodes columns in dataframe based on the column profile """
    encoder_dict = {}
    for column, col_type in profile.items():
        if col_type == 'nominal':
            encoder = OneHotEncoder()
            transformed = encoder.fit_transform(df[[column]]).toarray()
            # Create a dataframe with encoded columns
            cols = [f"{column}_{cat}" for cat in encoder.categories_[0]]
            encoded_df = pd.DataFrame(transformed, columns=cols)
            df = pd.concat([df.drop(column, axis=1), encoded_df], axis=1)
            encoder_dict[column] = encoder
        elif col_type == 'ordinal':
            encoder = LabelEncoder()
            df[column] = encoder.fit_transform(df[column])
            encoder_dict[column] = encoder
    return df, encoder_dict


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'csv'

if __name__ == '__main__':
    app.run(debug=True)
