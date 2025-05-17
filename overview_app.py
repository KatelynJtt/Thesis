from flask import Blueprint, render_template, jsonify, request, current_app
import pandas as pd
import numpy as np
import os


overview_bp = Blueprint('overview', __name__, template_folder='blueprints/templates/overview', static_folder='blueprints/static')

@overview_bp.route('/')
def overview():
    filename = request.args.get('file')
    if filename:
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(file_path):
            return render_template('overview.html', filename=filename)
    return render_template('overview.html', filename=None)


@overview_bp.route('/process_file')
def process_file():
    filename = request.args.get('file')
    if filename:
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            data = {
                'row_count': int(len(df)),
                'column_count': int(len(df.columns)),
                'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
                'missing_count': int(df.isnull().sum().sum()),
                'duplicate_count': int(df.duplicated().sum()),
                'columns': list(df.columns),
                'head_data': df.head().to_html(classes='table table-striped'),
                'sample_data': df.sample(5).to_html(classes='table table-striped')
            }
            return jsonify({'success': True, 'data': data})
    return jsonify({'success': False, 'error': 'File not found or invalid'})

@overview_bp.route('/all-columns/<filename>')
def all_columns(filename):
    file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    df = pd.read_csv(file_path)
    
    all_columns_data = {
        'memory_usage': df.memory_usage(deep=True).sum(),
        'missing_count': df.isnull().sum().to_dict(),
        'duplicate_count': df.duplicated().sum(),
        'dtypes': df.dtypes.astype(str).to_dict()
    }
    return jsonify(all_columns_data)

@overview_bp.route('/correlation/<filename>')
def correlation(filename):
    file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    df = pd.read_csv(file_path)
    
    # Get only numeric columns
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    correlation = numeric_df.corr()
    
    return jsonify({
        'correlation': correlation.values.tolist(),
        'columns': correlation.columns.tolist()
    })

@overview_bp.route('/distribution/<filename>/<column>')
def distribution(filename, column):
    file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    df = pd.read_csv(file_path)
    
    if pd.api.types.is_numeric_dtype(df[column]):
        return jsonify({
            'distribution': df[column].dropna().tolist(),
            'quartiles': df[column].quantile([0.25, 0.5, 0.75]).tolist()
        })
    else:
        value_counts = df[column].value_counts().head(10)
        return jsonify({
            'categories': value_counts.index.tolist(),
            'counts': value_counts.values.tolist()
        })

@overview_bp.route('/column-details/<column>')
def get_column_details(column):
    filename = request.args.get('file')
    if filename:
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            
            if column == 'All':
                return jsonify({
                    'type': 'all',
                    'details': {
                        'total_columns': int(len(df.columns)),
                        'numeric_columns': int(len(df.select_dtypes(include=[np.number]).columns)),
                        'categorical_columns': int(len(df.select_dtypes(include=['object']).columns)),
                        'dtypes': df.dtypes.astype(str).to_dict()
                    }
                })
            
            col_data = df[column]
            details = {
                'type': 'numeric' if np.issubdtype(col_data.dtype, np.number) else 'categorical',
                'missing_values': int(col_data.isnull().sum()),
                'unique_values': int(col_data.nunique())
            }
            
            if details['type'] == 'numeric':
                details.update({
                    'mean': float(col_data.mean()),
                    'median': float(col_data.median()),
                    'std': float(col_data.std()),
                    'min': float(col_data.min()),
                    'max': float(col_data.max()),
                    'skewness': float(col_data.skew()),
                    'kurtosis': float(col_data.kurtosis()),
                    'q1': float(col_data.quantile(0.25)),
                    'q3': float(col_data.quantile(0.75)),
                    'iqr': float(col_data.quantile(0.75) - col_data.quantile(0.25)),
                    'mode': float(col_data.mode().iloc[0]),
                    'variance': float(col_data.var())
                })
            else:
                details.update({
                    'total_values': len(col_data),
                    'value_counts': {str(k): int(v) for k, v in col_data.value_counts().head().items()}
                })
            
            return jsonify(details)
    return jsonify({'error': 'Invalid request'})

@overview_bp.route('/column-values/<column>')
def get_column_values(column):
    filename = request.args.get('file')
    if filename:
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            values = df[column].fillna('NA').tolist()
            return jsonify({'values': values})
    return jsonify({'error': 'Invalid request'})

