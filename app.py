from flask import Flask, render_template, request, redirect, url_for, jsonify, session
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import os
import plotly
import plotly.graph_objs as go
import json

app = Flask(__name__)

from logging.handlers import RotatingFileHandler

#-------------------------------------------------------Log functions for debugging specifics
@app.errorhandler(500)
def internal_error(error):
    app.logger.error('Server Error: %s', (error))
    return "500 error", 500

@app.errorhandler(Exception)
def unhandled_exception(e):
    app.logger.error('Unhandled Exception: %s', (e))
    return "500 error", 500

#---------------------------------------------------Display Functions
@app.route('/', methods=['GET', 'POST'])
def display():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            df = pd.read_csv(file)
            db_dir = './database'
            if not os.path.exists(db_dir):
                os.makedirs(db_dir)
            db_name = os.path.splitext(file.filename)[0] + '_database.db'
            table_name = os.path.splitext(file.filename)[0] + '_table'
            db_path = os.path.join(db_dir, db_name)
            engine = create_engine('sqlite:///' + db_path)
            df.to_sql(table_name, engine, if_exists='replace')
            # Store the db_name and table_name in the session data
            session['db_name'] = db_name
            session['table_name'] = table_name
            return render_template('display.html', tables=[df.to_html(classes='data')], titles=df.columns.values)
        else:
            return "File type is incorrect. Please upload a .csv file."
    return render_template('display.html')

#------------------------------------------------------------------Upload Function
@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file:
        df = pd.read_csv(file)
        num_columns = df.select_dtypes(include=np.number).columns.tolist()
        cat_columns = df.select_dtypes(exclude=np.number).columns.tolist()
        return render_template('display.html', tables=[df.to_html(classes='data')], num_columns=num_columns, cat_columns=cat_columns)
    return redirect(url_for('index'))

#-----------------------------------------------------------------Get Columns
@app.route('/columns', methods=['GET'])
def columns():
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
