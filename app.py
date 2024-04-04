from flask import Flask, render_template, request
import pandas as pd
from sqlalchemy import create_engine
import os

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

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'csv'
