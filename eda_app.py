from flask import Blueprint, render_template, request, jsonify, Flask, current_app
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO, StringIO
import base64
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import anderson
from werkzeug.utils import secure_filename
import os

#Create a Blueprint for the EDA app
eda_bp = Blueprint('eda', __name__, template_folder='blueprints/templates/eda', static_folder='blueprints/static')

matplotlib.use('Agg')
# Dictionary to store data from multiple files
file_data = {}

# ROUTES ##########################################################################################
# Define the route for the EDA page
@eda_bp.route('/')
def eda():
    filename = request.args.get('file')
    if filename:
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(file_path):
            # File exists, we can process it
            return render_template('eda.html', filename=filename)
    # If no file is selected or file doesn't exist, render the template without a file
    return render_template('eda.html', filename=None)

@eda_bp.route('/process_file')
def process_file():
    filename = request.args.get('file')
    if filename:
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            file_data[filename] = df
            data = df.to_dict(orient='records')
            return jsonify({'success': True, 'data': data})
    return jsonify({'success': False, 'error': 'File not found or invalid'})

@eda_bp.route('/show_dataset')
def show_dataset():
    filename = request.args.get('file')
    if filename in file_data:
        return jsonify(file_data[filename].head(100).to_dict(orient='records'))
    return jsonify({'error': 'No data uploaded for this file'})

@eda_bp.route('/missing_values')
def get_missing_values():
    filename = request.args.get('file')
    if filename not in file_data:
        return jsonify({'error': 'No data uploaded for this file'})
    
    df = file_data[filename]
    missing_values = df.isnull().sum()
    missing_data = pd.DataFrame({'Feature': missing_values.index, 'Missing Count': missing_values.values})
    missing_data = missing_data[missing_data['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
    
    return jsonify(missing_data.to_dict(orient='records'))

@eda_bp.route('/outliers')
def get_outliers():
    filename = request.args.get('file')
    if filename not in file_data:
        return jsonify({'error': 'No data uploaded for this file'})
    
    df = file_data[filename]
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    outlier_dict = {}
    
    for column in numerical_features:
        value = df[column]
        result = anderson(value)
        is_norm = result.statistic > result.critical_values[2]
        
        if is_norm:
            outliers = detect_outlier_zscore(value, 3)
        else:
            outliers = detect_outlier_iqr(value)
        
        if not outliers.empty:
            outlier_dict[column] = outliers.to_dict()
    
    return jsonify(outlier_dict)

def detect_outlier_zscore(df_col, threshold=3):
    z_score = (df_col - np.mean(df_col)) / np.std(df_col)
    return df_col[np.abs(z_score) > threshold]

def detect_outlier_iqr(df_col):
    q1 = np.percentile(df_col, 25)
    q3 = np.percentile(df_col, 75)
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    return df_col[(df_col < lower_bound) | (df_col > upper_bound)]

## GET COLUMNS FOR ANALYSIS ##############################
@eda_bp.route('/get_columns')
def get_columns():
    filename = request.args.get('file')
    if filename not in file_data:
        return jsonify({'error': 'No data uploaded for this file'})
    
    df = file_data[filename]
    return jsonify({'columns': df.columns.tolist()})

## UNIVARIATE ANALYSIS ##############################
@eda_bp.route('/univariate', methods=['POST'])
def univariate_analysis():
    filename = request.args.get('file')
    feature = request.json['feature']
    
    if filename not in file_data:
        return jsonify({'error': 'No data uploaded for this file'})
    
    df = file_data[filename]
    
    fig, ax = plt.subplots(figsize=(6, 4))
    if df[feature].dtype == 'object':
        sns.countplot(x=feature, data=df, ax=ax)
    else:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        df[feature].hist(grid=False, ax=ax[0])
        sns.boxplot(orient='h', x=df[feature], ax=ax[1])
    
    fig.suptitle(f'Univariate Analysis for {feature}')
    
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    
    return jsonify({'plot_url': plot_url})

## BIVARIATE ANALYSIS ##############################
@eda_bp.route('/bivariate', methods=['POST'])
def bivariate_analysis():
    filename = request.args.get('file')
    feature_1 = request.json['feature_1']
    feature_2 = request.json['feature_2']
    
    if filename not in file_data:
        return jsonify({'error': 'No data uploaded for this file'})
    
    df = file_data[filename]
    
    fig, ax = plt.subplots(figsize=(6, 4))
    if df[feature_1].dtype == 'object' and df[feature_2].dtype == 'object':
        crosstab = pd.crosstab(df[feature_1], df[feature_2])
        crosstab.plot(kind='bar', stacked=True, ax=ax)
    elif df[feature_1].dtype != 'object' and df[feature_2].dtype != 'object':
        sns.scatterplot(x=feature_1, y=feature_2, data=df, ax=ax)
        sns.regplot(x=feature_1, y=feature_2, data=df, scatter=False, ax=ax)
    else:
        sns.boxplot(x=feature_1, y=feature_2, data=df, ax=ax)
    
    plt.title(f'Bivariate Analysis: {feature_1} vs {feature_2}')
    
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    
    return jsonify({'plot_url': plot_url})

## MULTIVARIATE ANALYSIS ##############################
@eda_bp.route('/multivariate', methods=['POST'])
def multivariate_analysis():
    filename = request.args.get('file')
    features = request.json['features']
    
    if filename not in file_data:
        return jsonify({'error': 'No data uploaded for this file'})
    
    df = file_data[filename]
    data = df[features]
    
    vif_plot_url, vif_data = calculate_and_plot_vif(data)
    corr_plot_url, corr_data = calculate_and_plot_correlation(data)
    
    return jsonify({
        'vif_plot_url': vif_plot_url,
        'vif_data': vif_data,
        'corr_plot_url': corr_plot_url,
        'corr_data': corr_data
    })

## PIE CHART ##############################
@eda_bp.route('/pie_charts')
def get_pie_charts():
    filename = request.args.get('file')
    if filename not in file_data:
        return jsonify({'error': 'No data uploaded for this file'})
    df = file_data[filename]
    missing_plot_url, outliers_plot_url = calculate_and_plot_pie_charts(df)
    return jsonify({
        'missing_plot_url': missing_plot_url,
        'outliers_plot_url': outliers_plot_url
    })

## VIF ##############################
@eda_bp.route('/vif', methods=['GET', 'POST'])
def get_vif():
    filename = request.args.get('file')
    if filename not in file_data:
        return jsonify({'error': 'No data uploaded for this file'})
    df = file_data[filename]
    if request.method == 'POST':
        features = request.json.get('features')
        data = df[features]
    else:
        data = df.select_dtypes(include=[np.number])
    plot_url, vif_data = calculate_and_plot_vif(data)
    return jsonify({'plot_url': plot_url, 'vif_data': vif_data})

## CORRELATION ##############################
@eda_bp.route('/correlation', methods=['GET', 'POST'])
def get_correlation():
    filename = request.args.get('file')
    if filename not in file_data:
        return jsonify({'error': 'No data uploaded for this file'})
    df = file_data[filename]
    if request.method == 'POST':
        features = request.json.get('features')
        data = df[features]
    else:
        data = df
    plot_url, corr_data = calculate_and_plot_correlation(data)
    return jsonify({'plot_url': plot_url, 'corr_data': corr_data})

## TABLE MANIPULATION ##############################
# --SORT-- #
@eda_bp.route('/sort_table', methods=['POST'])
def sort_table():
    filename = request.json['file']
    feature = request.json['feature']
    ascending = request.json['ascending']
    
    if filename not in file_data:
        return jsonify({'error': 'No data uploaded for this file'})
    
    df = file_data[filename]
    sorted_data = df.sort_values(by=feature, ascending=ascending)
    return jsonify(sorted_data.head(100).to_dict(orient='records'))

# --FILTER-- #
@eda_bp.route('/filter_table', methods=['POST'])
def filter_table():
    filename = request.json['file']
    feature = request.json['feature']
    filter_value = request.json['filter_value']
    
    if filename not in file_data:
        return jsonify({'error': 'No data uploaded for this file'})
    
    df = file_data[filename]
    filtered_data = df[df[feature] == filter_value]
    return jsonify(filtered_data.head(100).to_dict(orient='records'))

# --GET UNIQUE VALUES-- #
@eda_bp.route('/get_unique_values', methods=['POST'])
def get_unique_values():
    filename = request.json['file']
    feature = request.json['feature']
    
    if filename not in file_data:
        return jsonify({'error': 'No data uploaded for this file'})
    
    df = file_data[filename]
    unique_values = df[feature].unique().tolist()
    return jsonify(unique_values)

## DISPLAY DATA INFO ##############################
@eda_bp.route('/dataset_info')
def get_dataset_info():
    filename = request.args.get('file')
    if filename not in file_data:
        return jsonify({'error': 'No data uploaded for this file'})
    
    df = file_data[filename]
    buffer = StringIO()
    df.info(buf=buffer)
    dataset_info = buffer.getvalue()
    
    return jsonify({'dataset_info': dataset_info})

## STATISTICAL SUMMARY ##############################
@eda_bp.route('/statistical_summary')
def get_statistical_summary():
    filename = request.args.get('file')
    if filename not in file_data:
        return jsonify({'error': 'No data uploaded for this file'})
    
    df = file_data[filename]
    summary = df.describe(include='all').T.reset_index()
    summary = summary.rename(columns={'index': 'Feature'})
    return jsonify(summary.to_dict(orient='records'))

## SAVE RESULTS ##############################
@eda_bp.route('/save_results', methods=['POST'])
def save_results():
    results = request.json['results']
    filename = request.json['filename']
    
    with open(f"{filename}.txt", "w") as f:
        f.write(results)
    
    return jsonify({'message': 'Results saved successfully'})

## RECOMENDATIONS ##############################
@eda_bp.route('/recommendations')
def get_recommendations():
    filename = request.args.get('file')
    if filename not in file_data:
        return jsonify({'error': 'No data uploaded for this file'})
    
    df = file_data[filename]
    recommendations = []

    # Missing values check
    missing_cols = df.columns[df.isnull().any()].tolist()
    if missing_cols:
        recommendations.append({
            'type': 'missing_values',
            'text': f"Consider handling missing values in columns: {', '.join(missing_cols)}",
            'features': missing_cols
        })

    # Categorical encoding check
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        recommendations.append({
            'type': 'categorical_encoding',
            'text': f"Consider encoding categorical variables: {', '.join(categorical_cols)}",
            'features': categorical_cols
        })

    # Skewness check for numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    skewed_cols = []
    for col in numerical_cols:
        if abs(df[col].skew()) > 1:
            skewed_cols.append(col)
    if skewed_cols:
        recommendations.append({
            'type': 'skewness',
            'text': f"Consider transforming skewed features: {', '.join(skewed_cols)}",
            'features': skewed_cols
        })

    # Outlier check
    outlier_cols = []
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        if df[((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))].shape[0] > 0:
            outlier_cols.append(col)
    if outlier_cols:
        recommendations.append({
            'type': 'outliers',
            'text': f"Consider handling outliers in: {', '.join(outlier_cols)}",
            'features': outlier_cols
        })

    # Correlation check
    if len(numerical_cols) > 1:
        corr_matrix = df[numerical_cols].corr()
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > 0.8:
                    high_corr_pairs.append(f"{corr_matrix.columns[i]} & {corr_matrix.columns[j]}")
        if high_corr_pairs:
            recommendations.append({
                'type': 'high_correlation',
                'text': f"Consider handling highly correlated features: {', '.join(high_corr_pairs)}",
                'pairs': high_corr_pairs
            })

    # VIF check
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numerical_features) >= 2:
        vif_plot_url, vif_data = calculate_and_plot_vif(df[numerical_features])
        if vif_data:
            high_vif = [feature for feature in vif_data if feature['VIF'] > 10]
            if high_vif:
                recommendations.append({
                    'type': 'high_vif',
                    'text': f"Consider removing or transforming features with high VIF: {', '.join([f['Feature'] for f in high_vif])}",
                    'features': [f['Feature'] for f in high_vif]
                })

    # Scale check
    scale_cols = []
    for col in numerical_cols:
        if df[col].std() > 100:
            scale_cols.append(col)
    if scale_cols:
        recommendations.append({
            'type': 'scaling',
            'text': f"Consider scaling features with large ranges: {', '.join(scale_cols)}",
            'features': scale_cols
        })

    return jsonify({'recommendations': recommendations})

@eda_bp.route('/apply_recommendation', methods=['POST'])
def apply_recommendation():
    data = request.json
    filename = data['file']
    rec_type = data['type']
    features = data['features']
    action = data['action']

    if filename not in file_data:
        return jsonify({'error': 'No data uploaded for this file'})

    df = file_data[filename]
    original_data = df.copy()
    changes = {}

    try:
        if rec_type == 'missing_values':
            if action == 'drop':
                df = df.dropna(subset=features)
            else:
                imputer = SimpleImputer(strategy=action)
                df[features] = imputer.fit_transform(df[features])
        elif rec_type == 'outliers':
            if action == 'iqr':
                for feature in features:
                    q1 = df[feature].quantile(0.25)
                    q3 = df[feature].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - (1.5 * iqr)
                    upper_bound = q3 + (1.5 * iqr)
                    df[feature] = df[feature].clip(lower_bound, upper_bound)
            elif action == 'zscore':
                for feature in features:
                    z_scores = np.abs((df[feature] - df[feature].mean()) / df[feature].std())
                    df = df[z_scores < 3]
        elif rec_type == 'scaling':
            if action == 'standard':
                scaler = StandardScaler()
            elif action == 'minmax':
                scaler = MinMaxScaler()
            df[features] = scaler.fit_transform(df[features])
        elif rec_type == 'high_vif':
            if action == 'remove':
                df = df.drop(columns=features)
            elif action == 'transform':
                for feature in features:
                    df[feature] = np.log1p(df[feature])
        elif rec_type == 'categorical':
            if action == 'onehot':
                encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
                encoded = encoder.fit_transform(df[features])
                encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names(features))
                df = pd.concat([df.drop(features, axis=1), encoded_df], axis=1)
            elif action == 'label':
                for feature in features:
                    df[feature] = LabelEncoder().fit_transform(df[feature])

        for feature in features:
            if feature in df.columns:
                changed_indices = (original_data[feature] != df[feature]).index
                changes[feature] = changed_indices.tolist()

        file_data[filename] = df
        return jsonify({'success': True, 'changes': changes})
    except Exception as e:
        return jsonify({'error': str(e)})


## EDA SUPPORT FUNCTIONS ##############################
def calculate_and_plot_pie_charts(df):
    missing_count = df.isnull().sum().sum()
    total_cells = df.size
    missing_percentage = (missing_count / total_cells) * 100

    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    outliers_count = sum(len(detect_outlier(df[col])) for col in numerical_features)
    outliers_percentage = (outliers_count / total_cells) * 100

    def create_pie_chart(percentages, labels, title):
        fig, ax = plt.subplots(figsize=(8, 6))
        wedges, texts, autotexts = ax.pie(percentages, autopct='%1.1f%%', startangle=0, pctdistance=0.75, colors=['#ff9999', '#66b3ff'])
        ax.legend(wedges, labels,
                title="Data",
                loc="center left",
                bbox_to_anchor=(1, 0, 0.5, 1),
                fontsize=16,
                title_fontsize=18)
        plt.setp(autotexts, size=20, weight="bold")
        ax.set_title(title, fontsize=20)
        
        plt.tight_layout()
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        plot_url = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        return plot_url

    missing_plot_url = create_pie_chart([missing_percentage, 100 - missing_percentage], 
                                        ['Missing' if missing_percentage > 0 else 'No Missing Values', 'Present'], 
                                        'Missing Values')
    outliers_plot_url = create_pie_chart([outliers_percentage, 100 - outliers_percentage], ['Outliers', 'Normal'], 'Outliers')

    return missing_plot_url, outliers_plot_url

def calculate_and_plot_vif(data, threshold=10):
    # Check if we have enough numerical data to calculate VIF
    if data.empty or data.shape[1] < 2:
        return None, []
        
    vif_data = pd.DataFrame()
    vif_data["Feature"] = data.columns
    vif_data["VIF"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
    vif_data = vif_data.sort_values(by='VIF', ascending=False)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(vif_data["Feature"], vif_data["VIF"], color='skyblue')
    ax.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold = {threshold}')
    ax.set_xlabel('Features', fontsize=10)
    ax.set_ylabel('VIF', fontsize=10)
    ax.set_title('Variance Inflation Factor (VIF)', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.legend(loc='best', fontsize=8)
    
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=200)
    buffer.seek(0)
    plot_url = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)
    
    return plot_url, vif_data.to_dict(orient='records')

def calculate_and_plot_correlation(data):
    corr_matrix = data.corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Matrix')
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=300)
    buffer.seek(0)
    plot_url = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)
    
    return plot_url, corr_matrix.to_dict()

def detect_outlier(df):
    if isinstance(df, pd.Series):
        df = df.to_frame()
    
    outlier_dict = {}
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    for column in numerical_features:
        value = df[column]
        result = anderson(value)
        is_norm = result.statistic > result.critical_values[2]
        
        if is_norm:
            outliers = detect_outlier_zscore(value, 3)
        else:
            outliers = detect_outlier_iqr(value)
        
        if not outliers.empty:
            outlier_dict[column] = outliers.to_dict()
    
    return outlier_dict

