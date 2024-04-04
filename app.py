from flask import Flask, render_template, request, redirect, url_for, jsonify
import pandas as pd
from sqlalchemy import create_engine
import seaborn as sns
import matplotlib.pyplot as plt
import os
from io import BytesIO
import base64

app = Flask(__name__)

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

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file:
        df = pd.read_csv(file)
        print(df.head())  # Print the first few rows of the DataFrame
        tables = [df.to_html(classes='data')]
        return render_template('display.html', tables=tables, columns=df.columns.tolist())
    return redirect(url_for('index'))

@app.route('/plot', methods=['POST'])
def generate_plot():
    data = request.get_json()
    df = pd.DataFrame(data)
    plt.figure(figsize=(10,6))
    sns.countplot(data=df, x=df.columns[0])
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    return jsonify({'plot_url': plot_url})

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'csv'

if __name__ == '__main__':
    app.run(debug=True)
