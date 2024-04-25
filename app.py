from flask import Flask, render_template, request, redirect, url_for, jsonify, session
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import os
import plotly
import plotly.graph_objs as go
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

#-------------------------------------------------------Log functions for debugging specifics
@app.errorhandler(Exception)
def handle_exception(e):
    app.logger.error('Unhandled Exception: %s', (e))
    return jsonify({'error': 'A server error occurred.', 'details': str(e)}), 500

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
@app.route('/', methods=['GET', 'POST'])
def display():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            get_data_and_columns(file)
            df = pd.read_sql_table(session['table_name'], create_engine('sqlite:///' + os.path.join('./database', session['db_name'])))
            return render_template('display.html', tables=[df.to_html(classes='data')], titles=df.columns.values, numeric_columns=session['numeric_columns'], categorical_columns=session['categorical_columns'])
        else:
            return "File type is incorrect. Please upload a .csv file."
    else:
        # When a GET request is made, clear the session and render the template without any tables or columns
        session.clear()
        return render_template('display.html')

#------------------------------------------------------------------Upload Function
@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file and allowed_file(file.filename):
        get_data_and_columns(file)
        db_path = os.path.join('./database', session['db_name'])
        engine = create_engine('sqlite:///' + db_path)
        df = pd.read_sql_table(session['table_name'], engine)
        return render_template('display.html', tables=[df.to_html(classes='data')], numeric_columns=session['numeric_columns'], categorical_columns=session['categorical_columns'])
    else:
        return jsonify({'error': 'File type is incorrect. Please upload a .csv file.'}), 400
        
#-------------------------------------------------------------JSONify Columns Function
# Define the '/jsonify_columns' endpoint
@app.route('/jsonify_columns', methods=['GET'])
def jsonify_columns():
    db_dir = './database'
    db_name = session.get('db_name')  # Get database name from session data
    table_name = session.get('table_name')  # Get table name from session data
    db_path = os.path.join(db_dir, db_name)
    engine = create_engine('sqlite:///' + db_path)
    df = pd.read_sql_table(table_name, engine)
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df.select_dtypes(include=[object]).columns.tolist()
    return jsonify(numeric_columns=numeric_columns, categorical_columns=categorical_columns)

#-----------------------------------------------------------------Plot Function (EDA)
@app.route('/plot', methods=['POST'])
def plot():
    if request.is_json:
        # Proceed if the request data is in JSON format
        data = request.get_json()
        column1 = data['column1']
        plot_type = data['plot_type']
        db_dir = './database'
        db_name = session.get('db_name')  # Get database name from session data
        table_name = session.get('table_name')  # Get table name from session data
        db_path = os.path.join(db_dir, db_name)
        engine = create_engine('sqlite:///' + db_path)
        df = pd.read_sql_table(table_name, engine)
        if plot_type == 'histogram':
            data = [go.Histogram(x=df[column1])]
        elif plot_type == 'box':
            data = [go.Box(y=df[column1])]
        elif plot_type == 'bar':
            data = [go.Bar(x=df[column1].value_counts().index, y=df[column1].value_counts().values)]
        elif plot_type == 'heatmap':
            data = [go.Heatmap(z=df.corr().values, x=df.columns, y=df.columns, colorscale='Viridis')]
        graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
        return render_template('plot.html', graphJSON=graphJSON)
    else:
        # Return an error response if the request data is not in JSON format
        return jsonify({"error": "Invalid data format. Expected JSON."}), 400



def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'csv'

if __name__ == '__main__':
    app.run(debug=True)
