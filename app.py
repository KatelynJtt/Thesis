from flask import Flask, render_template, request, jsonify, session, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO
import os
from werkzeug.utils import secure_filename
import logging
logging.getLogger('rasterio').setLevel(logging.WARNING)
import shutil
import pandas as pd
from datetime import datetime
import openpyxl
import matplotlib
matplotlib.use('Agg')

# FLASK BLUEPRINTS
from overview_app import overview_bp
from preprocessing_app import preprocessing_bp
from eda_app import eda_bp
#from geoimaging_app import geoimaging_bp >> Moved to main function to avoid circular import
from machinelearning_app import machinelearning_bp, setup_folders
from geoimaging_app import geoimaging_bp, init_socketio, setup_folders_gi, set_socketio

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Add this line for session support
CORS(app)  # This will allow requests from any origin

# Create the SocketIO instance
print("Creating SocketIO instance...")
socketio = SocketIO()
socketio.init_app(app)

# Set the instance in geoimaging_app
print("Setting the SocketIO instance in geoimaging_app...")
set_socketio(socketio)

# Initialize the socket events
print("Initializing SocketIO...")
init_socketio(socketio)


# UPLOAD FOLDER
app.config['UPLOAD_FOLDER'] = 'uploads'
# FLASK REGISTER BLUEPRINTS ----------------- this acts as first part of routes for each blueprint
app.register_blueprint(overview_bp, url_prefix='/overview')
app.register_blueprint(preprocessing_bp, url_prefix='/preprocessing')
app.register_blueprint(eda_bp, url_prefix='/eda')
app.register_blueprint(machinelearning_bp, url_prefix='/machinelearning')
app.register_blueprint(geoimaging_bp, url_prefix='/geoimaging')


def clear_uploads_folder():
    folder = app.config['UPLOAD_FOLDER']
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

print("Clearing uploads folder...")
clear_uploads_folder()
# Create Machine Learning Directories
setup_folders(app)
# Create Geoimaging Directories
setup_folders_gi(app)

@app.route('/')
def index():
    return render_template('base.html', content=render_template('home.html'), active_tab='home')

@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/get_files')
def get_files():
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    return jsonify({'files': files})

@app.route('/get_selected_file')
def get_selected_file():
    selected_file = session.get('selected_file')
    return jsonify({'filename': selected_file})

@app.route('/upload', methods=['POST'])
def upload_file():
    print("Files in request:", request.files)
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'})
    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return jsonify({'success': True, 'filename': filename})

@app.route('/file_info/<filename>')
def file_info(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(file_path):
        stats = os.stat(file_path)
        file_info = {
            'name': filename,
            'size': f"{stats.st_size / 1024:.2f} KB",
            'modified': datetime.fromtimestamp(stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
            'type': 'Excel' if filename.endswith('.xlsx') else 'CSV'
        }
        
        try:
            # Try reading as Excel first
            df = pd.read_excel(file_path, engine='openpyxl')
        except:
            try:
                # If Excel fails, try reading as CSV
                df = pd.read_csv(file_path, sep=None, engine='python')
            except Exception as e:
                return jsonify({'error': f'Unable to read file: {str(e)}'}), 400
            
        data = df.head(10).to_html(classes='table table-striped', index=False)
        return jsonify({'file_info': file_info, 'data': data})
    else:
        return jsonify({'error': 'File not found'}), 404

@app.route('/delete/<filename>', methods=['POST'])
def delete_file(filename):
    logging.info(f"Attempting to delete file: {filename}")
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    logging.info(f"Full file path: {file_path}")
    
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logging.info(f"Successfully deleted file: {filename}")
            return jsonify({'success': True})
        else:
            logging.warning(f"File not found: {filename}")
            return jsonify({'success': False, 'error': 'File not found'})
    except Exception as e:
        logging.error(f"Error deleting file {filename}: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/convert_excel')
def convert_excel():
    filename = request.args.get('filename')
    excel_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    csv_filename = filename.replace('.xlsx', '.csv')
    csv_path = os.path.join(app.config['UPLOAD_FOLDER'], csv_filename)
    
    try:
        # First try reading as CSV
        df = pd.read_csv(excel_path, sep=None, engine='python')
    except:
        # If CSV reading fails, try Excel format
        try:
            df = pd.read_excel(excel_path, engine='openpyxl')
        except:
            return jsonify({
                'success': False,
                'message': 'Unable to read file format'
            }), 400
    
    # Save as CSV
    df.to_csv(csv_path, index=False)
    
    return jsonify({
        'success': True,
        'csvFilename': csv_filename,
        'message': 'File converted successfully'
    })

@app.route('/download_file/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)




if __name__ == '__main__':
    print("Starting the application...")
    print("Registering Geoimaging Blueprint...")
    socketio.run(app, debug=True)

