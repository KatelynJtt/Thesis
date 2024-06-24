from flask import Flask, render_template, request, redirect, url_for, jsonify, session, send_from_directory
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import mpld3
import seaborn as sns
import os
import plotly
import plotly.graph_objs as go
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import json
import logging
from logging.handlers import RotatingFileHandler
import rasterio
import geopandas as gpd

from image_utils import RGB2Dataset
from eda_utils import create_fig_num, create_fig_cat, corr_heatmap



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

@app.route('/home')
def home():
    # Render the home.html template or generate the HTML content dynamically
    return render_template('home.html')

@app.route('/datasummary')
def data_summary():
    db_name = session.get('db_name')
    table_name = session.get('table_name')
    numeric_columns = session.get('numeric_columns', [])
    categorical_columns = session.get('categorical_columns', [])

    if db_name and table_name:
        db_path = os.path.join('./database', db_name)
        engine = create_engine('sqlite:///' + db_path)
        df = pd.read_sql_table(table_name, engine)
        return render_template('data_summary.html', tables=[df.to_html(classes='data')], titles=df.columns.values, numeric_columns=numeric_columns, categorical_columns=categorical_columns)
    else:
        return render_template('data_summary.html')

@app.route('/dataanalysis')
def data_analysis():
    db_name = session.get('db_name')
    table_name = session.get('table_name')
    numeric_columns = session.get('numeric_columns', [])
    categorical_columns = session.get('categorical_columns', [])

    if db_name and table_name:
        # A CSV file has been uploaded
        return render_template('data_analysis.html', numeric_columns=numeric_columns, categorical_columns=categorical_columns)
    else:
        # No CSV file has been uploaded
        return render_template('no_file_upload.html', page_name='Data Analysis', numeric_columns=[], categorical_columns=[])

#--------------------------------------------------------------------------------------------------------------------------------#### GEO-IMAGING
@app.route('/geoimaging', methods=['GET', 'POST'])
def geo_imaging():
    if request.method == 'POST':
        # Handle form submission
        action = request.form.get('action')

        # RGB processing
        if action.startswith('rgb_'):
            rgb_image = request.files.get('rgb_image')
            rgb_shapefile = request.files.get('rgb_shapefile')
            polygon_id_column = request.form.get('polygon_id_column')
            geometry_column = request.form.get('geometry_column')
            output_dir = request.form.get('output_dir')

            if rgb_image and rgb_shapefile:
                # Process RGB dataset
                filename = rgb_image.filename
                gdf = gpd.read_file(rgb_shapefile)
                with rasterio.open(rgb_image) as src:
                    dataset = RGB2Dataset(src, gdf, output_dir, filename)

                if action == 'rgb_geotiff':
                    dataset.clip_rasterio_shape()
                elif action == 'rgb_vi_histogram':
                    plot_index = request.form.get('plot_index')
                    dataset.visualization_plot(int(plot_index), True)
                elif action == 'rgb_clip_image':
                    dataset.clip_rasterio_shape()
                elif action == 'rgb_auto_extraction':
                    target_df = request.form.get('target_df')
                    if target_df:
                        dataset.dataset_extraction_auto(target_df)
                elif action == 'rgb_manual_extraction':
                    dataset.manu_extraction_window_rgb()

        # Multispectral processing
        elif action.startswith('ms_'):
            rband = request.files.get('rband')
            gband = request.files.get('gband')
            bband = request.files.get('bband')
            reband = request.files.get('reband')
            nirband = request.files.get('nirband')
            ms_shapefile = request.files.get('ms_shapefile')
            ms_polygon_id_column = request.form.get('ms_polygon_id_column')
            ms_geometry_column = request.form.get('ms_geometry_column')
            ms_output_dir = request.form.get('ms_output_dir')

            if rband and gband and bband and reband and nirband and ms_shapefile:
                # Process Multispectral dataset
                with rasterio.open(rband) as src_r, \
                     rasterio.open(gband) as src_g, \
                     rasterio.open(bband) as src_b, \
                     rasterio.open(reband) as src_re, \
                     rasterio.open(nirband) as src_nir:
                    gdf = gpd.read_file(ms_shapefile)
                    gdf = gdf.rename(columns={ms_polygon_id_column: "Plot_ID"})
                    gdf = gdf.rename(columns={ms_geometry_column: "geometry"})
                    dataset = MS2Dataset(src_r, src_g, src_b, src_re, src_nir, gdf, ms_output_dir, False)

                if action == 'ms_geotiff':
                    dataset.clip_rasterio_shape()
                elif action == 'ms_vi_histogram':
                    plot_index = request.form.get('plot_index')
                    dataset.visualization_plot(int(plot_index), True, None)
                elif action == 'ms_clip_image':
                    dataset.clip_rasterio_shape()
                elif action == 'ms_auto_extraction':
                    target_df = request.form.get('target_df')
                    if target_df:
                        dataset.dataset_extraction_auto(target_df)
                elif action == 'ms_manual_extraction':
                    dataset.manu_extraction_window_ms()

        # Return a success message or redirect to another route
        return jsonify({'message': 'Files processed successfully'})

    # Render the geo_imaging.html template
    return render_template('geo_imaging.html')

@app.route('/calculate-cover', methods=['POST'])
def calculate_cover():
    input_data = request.get_json()
    result = RGB2Dataset.calculate_canopy_cover(input_data)
    return jsonify(result)

@app.route('/calculate-vi', methods=['POST'])
def calculate_vi():
    input_data = request.get_json()
    red = input_data['red']
    green = input_data['green']
    blue = input_data['blue']
    selected_vi = input_data.get('selected_vi', None)
    result = RGB2Dataset.calculate_vi(red, green, blue, selected_vi)
    return jsonify(result)

@app.route('/extract-dataset', methods=['POST'])
def extract_dataset():
    # Assume 'target_data' is the key for the target DataFrame file
    target_df_file = request.files['target_data']
    target_df_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(target_df_file.filename))
    target_df_file.save(target_df_path)
    target_df = pd.read_excel(target_df_path)

    # Instantiate the class and call the method
    # Assume 'image' is the key for the image file
    image_file = request.files['image']
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(image_file.filename))
    image_file.save(image_path)

    # Open the image with rasterio
    with rasterio.open(image_path) as src:
        # Assume 'shapefile' is the key for the shapefile
        shapefile = request.files['shapefile']
        gdf = gpd.read_file(shapefile.stream)  # Read shapefile
        rgb_dataset = RGB2Dataset(src=src, gdf=gdf, output_dir=app.config['UPLOAD_FOLDER'], filename=image_file.filename)
        excel_filepath = rgb_dataset.dataset_extraction_auto(target_df)

    return send_from_directory(directory=os.path.dirname(excel_filepath), filename=os.path.basename(excel_filepath), as_attachment=True)

@app.route('/clip-shape', methods=['POST'])
def clip_shape():
    image_file = request.files['image']
    shapefile = request.files['shapefile']
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(image_file.filename))
    shapefile_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(shapefile.filename))
    image_file.save(image_path)
    shapefile.save(shapefile_path)

    # Instantiate the class and call the method
    rgb_dataset = RGB2Dataset(src=image_path, gdf=shapefile_path, output_dir=app.config['UPLOAD_FOLDER'], filename=image_file.filename)
    rgb_dataset.clip_rasterio_shape()

    return jsonify({"message": "Image clipped successfully"})

@app.route('/visualize-plot', methods=['POST'])
def visualize_plot():
    plot_index = int(request.form['plot_index'])
    hist_or_cc = request.form['hist_or_cc'].lower() == 'true'

    # Instantiate the class and call the method
    rgb_dataset = RGB2Dataset(...)  # Provide the necessary arguments
    fig_path = rgb_dataset.visualization_plot(plot_index, hist_or_cc)

    return send_from_directory(directory=os.path.dirname(fig_path), filename=os.path.basename(fig_path), as_attachment=True)

@app.route('/visualize-shpfile', methods=['GET'])
def visualize_shpfile():
    # Instantiate the class and call the method
    rgb_dataset = RGB2Dataset(...)  # Provide the necessary arguments
    fig_path = rgb_dataset.visualization_shpfile()

    return send_from_directory(directory=os.path.dirname(fig_path), filename=os.path.basename(fig_path), as_attachment=True)

@app.route('/check-original', methods=['GET'])
def check_original():
    # Instantiate the class and call the method
    rgb_dataset = RGB2Dataset(...)  # Provide the necessary arguments
    fig_path = rgb_dataset.check_original()

    return send_from_directory(directory=os.path.dirname(fig_path), filename=os.path.basename(fig_path), as_attachment=True)

@app.route('/check-clipped', methods=['GET'])
def check_clipped():
    # Instantiate the class and call the method
    rgb_dataset = RGB2Dataset(...)  # Provide the necessary arguments
    fig_path = rgb_dataset.check_clipped()

    return send_from_directory(directory=os.path.dirname(fig_path), filename=os.path.basename(fig_path), as_attachment=True)
#--------------------------------------------------------------------------------------------------------------------------------------End of Geo Routes

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


#------------------------------------------------------------------Upload Function
@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file and allowed_file(file.filename):
        get_data_and_columns(file)
        db_path = os.path.join('./database', session['db_name'])
        engine = create_engine('sqlite:///' + db_path)
        df = pd.read_sql_table(session['table_name'], engine)
        success_message = "File uploaded successfully!"
        return render_template('index.html', success_message=success_message)
    else:
        error_message = 'File type is incorrect. Please upload a .csv file.'
        return render_template('index.html', error_message=error_message)

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
    if request.is_json:
        data = request.get_json()
        plot_type = data['plot_type']
        db_dir = './database'
        db_name = session.get('db_name')
        table_name = session.get('table_name')
        db_path = os.path.join(db_dir, db_name)
        engine = create_engine('sqlite:///' + db_path)
        df = pd.read_sql_table(table_name, engine)

        if plot_type == 'univariate':
            column = data['column']
            if df[column].dtype == 'object':
                fig = create_fig_cat(column, df)
            else:
                fig = create_fig_num(column, df[column])

            # Convert the figure to HTML
            html_fig = mpld3.fig_to_html(fig)
            return jsonify({'html_fig': html_fig})

        elif plot_type == 'bivariate':
            column1 = data['column1']
            column2 = data['column2']

            # Perform bivariate analysis (e.g., scatter plot, box plot, etc.)
            fig, ax = plt.subplots(figsize=(8, 6))
            if df[column1].dtype == 'object' or df[column2].dtype == 'object':
                sns.boxplot(x=column1, y=column2, data=df, ax=ax)
            else:
                sns.scatterplot(x=column1, y=column2, data=df, ax=ax)

            # Convert the figure to HTML
            html_fig = mpld3.fig_to_html(fig)
            return jsonify({'html_fig': html_fig})

        elif plot_type == 'multivariate':
            columns = data['columns']

            # Perform multivariate analysis (e.g., correlation heatmap)
            fig, ax = plt.subplots(figsize=(8, 6))
            corr = df[columns].corr()
            sns.heatmap(corr, annot=True, ax=ax)

            # Convert the figure to HTML
            html_fig = mpld3.fig_to_html(fig)
            return jsonify({'html_fig': html_fig})

    else:
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
