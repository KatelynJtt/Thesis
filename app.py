from flask import Flask, render_template, request, redirect, url_for, jsonify
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import seaborn as sns
import matplotlib.pyplot as plt
import os
from io import BytesIO
import base64
import logging

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

#-----------------------------------------------------------------Plot Function (EDA)
@app.route('/plot', methods=['POST'])
def generate_plot():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        df = pd.DataFrame(data)
        plot_type = data.get('plot_type')
        column = data.get('column')

        if plot_type not in ['hist', 'box', 'bar', 'heatmap']:
            return jsonify({'error': 'Invalid plot type'}), 400

        if plot_type in ['hist', 'box', 'bar'] and column not in df.columns:
            return jsonify({'error': 'Invalid column name'}), 400

        if plot_type == 'hist':
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data=df, x=column, ax=ax)
        elif plot_type == 'box':
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=df, x=column, ax=ax)
        elif plot_type == 'bar':
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.countplot(data=df, x=column, ax=ax)
        elif plot_type == 'heatmap':
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    return jsonify({'plot_url': plot_url})

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'csv'

if __name__ == '__main__':
    app.run(debug=True)
