from flask import Blueprint, Flask, render_template, request, jsonify, session, render_template_string, send_file, send_from_directory, after_this_request, current_app
import json
import os
import shutil
import joblib
import time
import pandas as pd
from werkzeug.utils import secure_filename

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import BaggingRegressor, StackingRegressor, VotingRegressor, BaggingClassifier, StackingClassifier, VotingClassifier
from sklearn.preprocessing import LabelEncoder

import zipfile
import base64
import csv
import io
import traceback

from ml_utils import regressor_dict, classifier_dict, default_param_grids, explain_model_shap, explain_model_lime, plot_regression_result, calculate_scores, param_tips, singleML_regression, singleML_classification, ensembleML_regression, ensembleML_classification


SPLIT_DATA_FOLDER = None
TRAINED_MODELS_FOLDER = None 
EXPLAIN_MODELS_FOLDER = None

# Create a Blueprint for the machine learning app
machinelearning_bp = Blueprint('machinelearning', __name__, template_folder='blueprints/templates/machinelearning', static_folder='blueprints/static')

# Don't need to use current_app.config['UPLOAD_FOLDER'] because we are using the global variable passed in from app.py
def setup_folders(app):
    print("Setting up ML folders...")
    global SPLIT_DATA_FOLDER, TRAINED_MODELS_FOLDER, EXPLAIN_MODELS_FOLDER
    BASE_UPLOAD_FOLDER = app.config['UPLOAD_FOLDER']
    
    SPLIT_DATA_FOLDER = os.path.join(BASE_UPLOAD_FOLDER, 'split_data')
    TRAINED_MODELS_FOLDER = os.path.join(BASE_UPLOAD_FOLDER, 'trained_models')
    EXPLAIN_MODELS_FOLDER = os.path.join(BASE_UPLOAD_FOLDER, 'explain_models')
    
    print(f"SPLIT_DATA_FOLDER set to: {SPLIT_DATA_FOLDER}")
    print(f"TRAINED_MODELS_FOLDER set to: {TRAINED_MODELS_FOLDER}")
    print(f"EXPLAIN_MODELS_FOLDER set to: {EXPLAIN_MODELS_FOLDER}")

    for folder in [SPLIT_DATA_FOLDER, TRAINED_MODELS_FOLDER, EXPLAIN_MODELS_FOLDER]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)
        print(f"Created folder: {folder}")

# ROUTES ##########################################################################################
# Define a route for the machine learning app
@machinelearning_bp.route('/')
def machinelearning():
    filename = request.args.get('file')
    if filename:
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(file_path):
            # File exists, we can process it
            return render_template('machinelearning.html', filename=filename)
    # If no file is selected or file doesn't exist, render the template without a file
    return render_template('machinelearning.html', filename=None)

@machinelearning_bp.route('/get_columns')
def get_columns():
    filename = request.args.get('file')
    if filename:
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        df = pd.read_csv(filepath)
        columns = df.columns.tolist()
        return jsonify({'columns': columns})
    return jsonify({'error': 'No file selected'}), 400

@machinelearning_bp.route('/select_features', methods=['POST'])
def select_features():
    data = request.json
    filename = data.get('filename')
    current_features = data.get('features')
    current_label = data.get('label')
    current_split = float(data.get('test_size', 0.3))

    filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    df = pd.read_csv(filepath)
    X = df[current_features]
    y = df[current_label]

    missing_values = X.isna().sum().sum() + y.isna().sum()
    if missing_values > 0:
        message = f'There are {missing_values} missing values in this dataset. Please upload a dataset without missing values.'
    elif X.select_dtypes(include=['object']).shape[1] > 0:
        message = 'There are non-numeric data in this dataset.'
    else:
        # Prepare for regression
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y, test_size=current_split, random_state=42)

        # Prepare for classification
        le = LabelEncoder()
        y_class = le.fit_transform(y)
        X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X, y_class, test_size=current_split, random_state=42)

        split_data_path = os.path.join(SPLIT_DATA_FOLDER, f'split_data_{int(time.time())}.joblib')
        joblib.dump({
            'regression': {
                'X_train': X_train_reg,
                'X_test': X_test_reg,
                'y_train': y_train_reg,
                'y_test': y_test_reg
            },
            'classification': {
                'X_train': X_train_class,
                'X_test': X_test_class,
                'y_train': y_train_class,
                'y_test': y_test_class,
                'label_encoder': le
            }
        }, split_data_path)

        session['split_data_path'] = split_data_path
        message = 'Features and label selected successfully. Data split into train and test sets.'

    return jsonify({'message': message, 'split_data_path': session.get('split_data_path')})


## EXPLAIN with SHAP / LIME ########################################################
@machinelearning_bp.route('/explain_shap', methods=['POST'])
def explain_shap():
    data = request.json
    models = data['models']
    ensemble_method = data['ensemble_method']
    
    print(f"Models passed to SHAP function: {models}")
    print(f"Ensemble method: {ensemble_method}")
    
    split_data = joblib.load(session['split_data_path'])
    
    if any(model in regressor_dict for model in models):
        data_set = split_data['regression']
        model_dict = regressor_dict
    elif any(model in classifier_dict for model in models):
        data_set = split_data['classification']
        model_dict = classifier_dict
    else:
        return jsonify({'error': 'Invalid model type'}), 400
    
    X_train, X_test = data_set['X_train'], data_set['X_test']
    
    with open(session['current_model_metadata_path'], 'r') as f:
        model_data = json.load(f)
    
    model_data['model_path'] = session['current_model_path']
    model_data['ensemble_method'] = ensemble_method
    
    shap_result = explain_model_shap(model_data, models, X_train, X_test, model_dict)
    
    if 'error' in shap_result:
        return jsonify(shap_result), 400
        
    filename = f"shap_plot_{int(time.time())}.png"
    filepath = os.path.join(EXPLAIN_MODELS_FOLDER, filename)
    
    with open(filepath, 'wb') as f:
        f.write(base64.b64decode(shap_result['plot_url']))
    
    return jsonify({'plot_url': shap_result['plot_url'], 'filename': filename})

@machinelearning_bp.route('/explain_lime', methods=['POST'])
def explain_lime():
    data = request.json
    models = data['models']
    ensemble_method = data['ensemble_method']
    
    print(f"Models passed to LIME function: {models}")
    print(f"Ensemble method: {ensemble_method}")
    
    split_data = joblib.load(session['split_data_path'])
    
    if any(model in regressor_dict for model in models):
        data_set = split_data['regression']
        model_dict = regressor_dict
    elif any(model in classifier_dict for model in models):
        data_set = split_data['classification']
        model_dict = classifier_dict
    else:
        return jsonify({'error': 'Invalid model type'}), 400
    
    X_train, X_test = data_set['X_train'], data_set['X_test']
    
    with open(session['current_model_metadata_path'], 'r') as f:
        model_data = json.load(f)
    
    model_data['model_path'] = session['current_model_path']
    model_data['ensemble_method'] = ensemble_method
    
    lime_result = explain_model_lime(model_data, models, X_train, X_test, model_dict)
    
    filename = f"lime_plot_{int(time.time())}.png"
    filepath = os.path.join(EXPLAIN_MODELS_FOLDER, filename)
    with open(filepath, 'wb') as f:
        f.write(base64.b64decode(lime_result['plot_url']))
    
    return jsonify({'plot_url': lime_result['plot_url'], 'filename': filename})

@machinelearning_bp.route('/download_explanation_plot', methods=['POST'])
def download_explanation_plot():
    data = request.json
    filename = data['filename']
    return send_from_directory(EXPLAIN_MODELS_FOLDER, filename, as_attachment=True)

## Setting Model Paramters ##########################################################################################################################
# Define metrics matching ml_utils.py
REGRESSION_METRICS = ['R-squared', 'MAE', 'MSE', 'RMSE', 'MAX Error', 'MAPE']
CLASSIFICATION_METRICS = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']

@machinelearning_bp.route('/get_regression_modal/')
def get_regression_modal():
    return jsonify({
        'model_type': 'regression',
        'metrics': REGRESSION_METRICS
    })

@machinelearning_bp.route('/get_classification_modal/')
def get_classification_modal():
    return jsonify({
        'model_type': 'classification',
        'metrics': CLASSIFICATION_METRICS
    })

@machinelearning_bp.route('/get_model_params/<model_name>')
def get_model_params(model_name):
    print(f"Fetching parameters for model: {model_name}")
    print(f"Route hit: /get_model_params/{model_name}")
    print(f"Available models in regressor_dict: {list(regressor_dict.keys())}")
    print(f"Available models in classifier_dict: {list(classifier_dict.keys())}")
    
    # Check both regressor and classifier dictionaries
    if model_name in regressor_dict:
        model_class = regressor_dict[model_name]
    elif model_name in classifier_dict:
        model_class = classifier_dict[model_name]
    else:
        return jsonify({'error': f'Model {model_name} not found'}), 404
        
    params = default_param_grids.get(model_name, {})
    print(f"Retrieved parameters for {model_name}: {params}")
    return jsonify(params)

@machinelearning_bp.route('/get_model_layout/<model_name>/<search_method>')
def get_model_layout(model_name, search_method):
    print(f"Fetching layout for {model_name} with search method {search_method}")
    param_grid = default_param_grids.get(model_name, {})
    processed_params = {}

    if search_method == 'none':
        for param, values in param_grid.items():
            if isinstance(values[0], str):
                processed_params[param] = {
                    'type': 'dropdown',
                    'options': values,
                    'default': values[0]
                }
            else:
                processed_params[param] = {
                    'type': 'input',
                    'default': values[0]
                }
    else:  # Grid Search or Random Search
        for param, values in param_grid.items():
            processed_params[param] = {
                'type': 'input',
                'value': ', '.join(map(str, values))
            }

    return jsonify(processed_params)

## Train and Predict ##########################################################################################################################
@machinelearning_bp.route('/train_and_predict', methods=['POST'])
def train_and_predict():
    print("Received train_and_predict request")
    data = request.json
    models = data['models']
    model_type = data['model_type']
    ensemble_method = data['ensemble_method']
    search_method = data['search_method']
    params = data['params']

    is_regression = model_type == "Regressor"

    try:
        split_data = joblib.load(session['split_data_path'])
        data_set = split_data['regression'] if is_regression else split_data['classification']
        X_train, X_test, y_train, y_test = data_set['X_train'], data_set['X_test'], data_set['y_train'], data_set['y_test']

        if ensemble_method != 'none':
            if is_regression:
                result = ensembleML_regression(ensemble_method, models, X_train, X_test, y_train, y_test)
            else:
                result = ensembleML_classification(ensemble_method, models, X_train, X_test, y_train, y_test)
        else:
            if is_regression:
                result = singleML_regression(models[0], X_train, X_test, y_train, y_test, params, search_method)
            else:
                result = singleML_classification(models[0], X_train, X_test, y_train, y_test, params, search_method)
        
        model = result.pop('model', None)
        
        model_data = {
            'model': models[0] if ensemble_method == 'none' else ensemble_method,
            'ensemble_method': ensemble_method,
            'params': params,
            'metrics': {
                'train': result['train_metrics'],
                'test': result['test_metrics']
            }
        }
        
        json_filename = f"{models[0] if ensemble_method == 'none' else ensemble_method}_metadata_{int(time.time())}.json"
        json_path = os.path.join(TRAINED_MODELS_FOLDER, json_filename)
        with open(json_path, 'w') as f:
            json.dump(model_data, f)

        if model:
            model_filename = f"{models[0] if ensemble_method == 'none' else ensemble_method}_model_{int(time.time())}.joblib"
            model_path = os.path.join(TRAINED_MODELS_FOLDER, model_filename)
            joblib.dump(model, model_path)
            session['current_model_path'] = model_path

        session['current_model_metadata_path'] = json_path

        return jsonify(result)

    except Exception as e:
        print(f"Error in train_and_predict: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

## Save Trained Model ########################################################################################################################
@machinelearning_bp.route('/download_trained_model')
def download_trained_model():
    model_path = session.get('current_model_path')
    if model_path and os.path.exists(model_path):
        filename = os.path.basename(model_path)
        return send_from_directory(TRAINED_MODELS_FOLDER, filename, as_attachment=True)
    else:
        return jsonify({'success': False, 'error': 'Model not found'}), 404

@machinelearning_bp.route('/serve_model_file/<filename>')
def serve_model_file(filename):
    return send_from_directory(TRAINED_MODELS_FOLDER, filename, as_attachment=True)

## Save Displayed Result #####################################################################################################################
@machinelearning_bp.route('/save_displayed_result', methods=['POST'])
def save_displayed_result():
    data = request.json
    train_score = data['trainScore']
    test_score = data['testScore']
    images = data['images']

    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Add CSV files
        csv_buffer = io.StringIO()
        writer = csv.writer(csv_buffer)
        writer.writerow(['Train Results'])
        writer.writerow(['Metric', 'Score'])
        writer.writerows(train_score)
        writer.writerow([])
        writer.writerow(['Test Results'])
        writer.writerow(['Metric', 'Score'])
        writer.writerows(test_score)
        zf.writestr('results.csv', csv_buffer.getvalue())

        # Add images - using img_data directly without splitting
        for img_name, img_data in images.items():
            zf.writestr(f'{img_name}.png', base64.b64decode(img_data))

    memory_file.seek(0)
    return send_file(
        memory_file,
        mimetype='application/zip',
        as_attachment=True,
        download_name='model_results.zip'
    )

## Help ######################################################################################################################################
@machinelearning_bp.route('/get_help', methods=['GET'])
def get_help():
    regressor = request.args.get('regressor')
    para_text = param_tips.get(regressor, {})
    help_text = '\n\n'.join([f'{key}: {value}' for key, value in para_text.items()])
    return jsonify({'help_text': help_text})

