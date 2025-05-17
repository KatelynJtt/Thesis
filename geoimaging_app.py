from flask import current_app, Blueprint, render_template, request, jsonify, Flask, send_from_directory, send_file, redirect, url_for
from flask_socketio import SocketIO
import io
import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt
import os
import rasterio
from rasterio.mask import mask
from matplotlib.cm import ScalarMappable
from rasterio.plot import show
from tqdm import tqdm
import base64
import threading
from queue import Queue
from datetime import datetime
import shutil
import time



def setup_folders_gi(app):
    print("Setting up GI folders...")
    BASE_UPLOAD_FOLDER = app.config['UPLOAD_FOLDER']
    
    # Create all directory paths
    directories = {
        'RGB_DIR': os.path.join(BASE_UPLOAD_FOLDER, 'rgb'),
        'MS_DIR': os.path.join(BASE_UPLOAD_FOLDER, 'ms'),
    }
    
    # Add subdirectories
    directories.update({
        'RGB_IMAGE_DIR': os.path.join(directories['RGB_DIR'], 'rgb_image'),
        'RGB_SHAPEFILE_DIR': os.path.join(directories['RGB_DIR'], 'rgb_shapefile'),
        'RGB_OUTPUT_DIR': os.path.join(directories['RGB_DIR'], 'output'),
        'MS_R_BAND_DIR': os.path.join(directories['MS_DIR'], 'r_band'),
        'MS_G_BAND_DIR': os.path.join(directories['MS_DIR'], 'g_band'),
        'MS_B_BAND_DIR': os.path.join(directories['MS_DIR'], 'b_band'),
        'MS_RE_BAND_DIR': os.path.join(directories['MS_DIR'], 're_band'),
        'MS_NIR_BAND_DIR': os.path.join(directories['MS_DIR'], 'nir_band'),
        'MS_SHAPEFILE_DIR': os.path.join(directories['MS_DIR'], 'ms_shapefile'),
        'MS_OUTPUT_DIR': os.path.join(directories['MS_DIR'], 'output')
    })
    
    # Create all directories
    for directory in directories.values():
        os.makedirs(directory, exist_ok=True)
    
    # Store the paths in the app config
    app.config['GI_DIRS'] = directories
    
    return directories


# Initialize the dataset objects
rgb_dataset = None
ms_dataset = None
crs_checked = False


# Function to save a file to the output directory
def save_file_to_output_dir(file_data, filename, output_dir):
    file_path = os.path.join(output_dir, filename)
    with open(file_path, 'wb') as file:
        file.write(file_data)

# Global socketio instance
socketio = None

def set_socketio(socket_instance):
    global socketio
    socketio = socket_instance

def init_socketio(socket_instance):
    @socket_instance.on('connect', namespace='/geoimaging')
    def handle_connect():
        print('Client connected')

    @socket_instance.on('update_output_files', namespace='/geoimaging')
    def update_output_files_container(file_type):
        gi_dirs = current_app.config['GI_DIRS']
        output_dir = gi_dirs['RGB_OUTPUT_DIR'] if file_type == 'rgb' else gi_dirs['MS_OUTPUT_DIR']
        files = os.listdir(output_dir)
        socket_instance.emit('update_output_files', {'files': files, 'fileType': file_type}, namespace='/geoimaging')

    # clipping completion handler
    @socket_instance.on('clipping_complete', namespace='/geoimaging')
    def handle_clipping_complete(data):
        socket_instance.emit('clipping_complete', data, namespace='/geoimaging')

    return socket_instance

# Create a Blueprint for the geoimaging app
geoimaging_bp = Blueprint('geoimaging', __name__, template_folder='blueprints/templates/geoimaging', static_folder='blueprints/static')
print("Initializing geoimaging app...")

@geoimaging_bp.record
def record_params(setup_state):
    geoimaging_bp.app = setup_state.app

# ROUTES ##########################################################################################
# Define a route for the geoimaging app
@geoimaging_bp.route('/')
def geoimaging():
    print("Rendering geoimaging.html")
    return render_template('geoimaging.html')

@geoimaging_bp.route('/get_rgb_output_files', methods=['GET'])
def get_rgb_output_files():
    gi_dirs = current_app.config['GI_DIRS']
    files = os.listdir(gi_dirs['RGB_OUTPUT_DIR'])
    return jsonify({'files': files})

@geoimaging_bp.route('/get_ms_output_files', methods=['GET'])
def get_ms_output_files():
    gi_dirs = current_app.config['GI_DIRS']
    files = os.listdir(gi_dirs['MS_OUTPUT_DIR'])
    return jsonify({'files': files})

## CREATE DATASETS ##########################################################################################################
def create_rgb_dataset(image_path, shapefile_paths):
    print("\n=== Creating RGB Dataset ===")
    try:
        print(f"Opening image: {image_path}")
        src = rasterio.open(image_path)
        print(f"Image CRS: {src.crs}")
        print(f"Image shape: {src.shape}")
        
        shp_file = next((path for path in shapefile_paths if path.endswith('.shp')), None)
        print(f"Using shapefile: {shp_file}")
        
        if not shp_file:
            print('Error: No .shp file found in the provided shapefile paths.')
            return None, False, 'No .shp file found'
            
        gdf = gpd.read_file(shp_file)
        print(f"Shapefile CRS: {gdf.crs}")
        print(f"Number of polygons: {len(gdf)}")
        
        file_name = os.path.splitext(os.path.basename(image_path))[0]
        print(f"Using filename: {file_name}")
        
        # Check for clipped images directory
        output_dir = current_app.config['GI_DIRS']['RGB_OUTPUT_DIR']
        clip_dir = os.path.join(current_app.config['GI_DIRS']['RGB_OUTPUT_DIR'], 'shape_clip_' + file_name)
        print(f"Checking clipped images directory: {clip_dir}")
        
        # Verify all plot images exist
        all_plots_clipped = True
        if os.path.exists(clip_dir):
            for plot_id in gdf['Plot_ID']:
                plot_file = os.path.join(clip_dir, f'{plot_id}.tif')
                if not os.path.exists(plot_file):
                    print(f"Missing clipped image for plot: {plot_id}")
                    all_plots_clipped = False
                    break
            print("All clipped images found" if all_plots_clipped else "Missing some clipped images")
        else:
            print("Clipped images directory not found")
            all_plots_clipped = False
        
        gdf = gdf.sort_values(by='Plot_ID', ascending=True)
        print("Sorted GeoDataFrame by Plot_ID")
        
        # Check the CRS
        crs_matched, crs_mismatch_message = check_crs_match(gdf, src)
        if crs_matched:
            print("CRS match confirmed")
            rgb_dataset = RGB2Dataset(
                src=src,
                gdf=gdf,
                filename=file_name,
                output_dir=output_dir,
                clip_dir=clip_dir
            )
            print("RGB2Dataset instance created successfully")
            return rgb_dataset, True, None
        else:
            print("CRS mismatch detected")
            print(f"Shapefile CRS {shp_file}: {gdf.crs}")
            print(f"Image CRS {image_path}: {src.crs}")
            return None, False, crs_mismatch_message

    except Exception as e:
        print(f'Error creating RGB2Dataset: {e}')
        return None, False, str(e)

def create_ms_dataset(band_paths, shapefile_paths):
    print("\n=== Creating MS Dataset ===")
    try:
        print("Opening band files...")
        src_r = rasterio.open(band_paths['r_band'])
        src_g = rasterio.open(band_paths['g_band'])
        src_b = rasterio.open(band_paths['b_band'])
        src_re = rasterio.open(band_paths['re_band'])
        src_nir = rasterio.open(band_paths['nir_band'])
        print("All band files opened successfully")
        print(f"Band CRS: {src_r.crs}")
        print(f"Band shape: {src_r.shape}")

        shp_file = next((path for path in shapefile_paths if path.endswith('.shp')), None)
        print(f"Using shapefile: {shp_file}")

        if not shp_file:
            print('Error: No .shp file found in the provided shapefile paths.')
            return None, False, 'No .shp file found'

        gdf = gpd.read_file(shp_file)
        print(f"Shapefile CRS: {gdf.crs}")
        print(f"Number of polygons: {len(gdf)}")
        
        # Check for clipped images directory
        output_dir = current_app.config['GI_DIRS']['MS_OUTPUT_DIR']
        clip_dir = os.path.join(current_app.config['GI_DIRS']['MS_OUTPUT_DIR'], 'shape_clip')
        print(f"Checking clipped images directory: {clip_dir}")
        
        # Verify all plot images exist for each band
        all_plots_clipped = True
        if os.path.exists(clip_dir):
            for band in ['red', 'green', 'blue', 're', 'nir']:
                band_dir = os.path.join(clip_dir, band)
                print(f"Checking {band} band directory: {band_dir}")
                if os.path.exists(band_dir):
                    for plot_id in gdf['Plot_ID']:
                        plot_file = os.path.join(band_dir, f'{plot_id}.tif')
                        if not os.path.exists(plot_file):
                            print(f"Missing clipped image for plot {plot_id} in {band} band")
                            all_plots_clipped = False
                            break
                else:
                    print(f"Missing band directory: {band_dir}")
                    all_plots_clipped = False
            print("All clipped images found" if all_plots_clipped else "Missing some clipped images")
        else:
            print("Clipped images directory not found")
            all_plots_clipped = False

        gdf = gdf.sort_values(by='Plot_ID', ascending=True)
        print("Sorted GeoDataFrame by Plot_ID")

        # Check the CRS
        crs_matched, crs_mismatch_message = check_crs_match(gdf, src_r)
        if crs_matched:
            print("CRS match confirmed")
            ms_dataset = MS2Dataset(
                src_r=src_r,
                src_g=src_g,
                src_b=src_b,
                src_re=src_re,
                src_nir=src_nir,
                gdf=gdf,
                output_dir=output_dir,
                clip_dir=clip_dir,
                is_stacked=False
            )
            print("MS2Dataset instance created successfully")
            return ms_dataset, True, None
        else:
            print("CRS mismatch detected")
            print(f"Shapefile CRS {shp_file}: {gdf.crs}")
            print(f"Image CRS: {src_r.crs}")
            return None, False, crs_mismatch_message

    except Exception as e:
        print(f'Error creating MS2Dataset: {e}')
        return None, False, str(e)


@geoimaging_bp.route('/create_dataset', methods=['POST'])
def create_dataset():
    print("\n=== Starting Dataset Creation ===")
    image_type = request.form.get('imageType')
    print(f"Image type: {image_type}")
    
    gi_dirs = current_app.config['GI_DIRS']
    print("Got GI directories")

    if image_type == 'rgb':
        print("Processing RGB dataset")
        rgb_image_files = os.listdir(gi_dirs['RGB_IMAGE_DIR'])
        rgb_shapefile_files = os.listdir(gi_dirs['RGB_SHAPEFILE_DIR'])
        print(f"Found {len(rgb_image_files)} image files and {len(rgb_shapefile_files)} shapefile files")

        if rgb_image_files and rgb_shapefile_files:
            image_path = os.path.join(gi_dirs['RGB_IMAGE_DIR'], rgb_image_files[0])
            filename = os.path.splitext(os.path.basename(image_path))[0]
            dataset_dir = os.path.join(gi_dirs['RGB_OUTPUT_DIR'], 'Excel_' + filename)
            shapefile_paths = [os.path.join(gi_dirs['RGB_SHAPEFILE_DIR'], f) for f in rgb_shapefile_files]
            
            print(f"Creating RGB dataset with image: {image_path}")
            print(f"Output directory: {dataset_dir}")
            
            global rgb_dataset
            rgb_dataset, crs_matched, crs_mismatch_message = create_rgb_dataset(image_path, shapefile_paths)
            
            if rgb_dataset:
                print("RGB dataset created successfully")
                rgb_dataset.clip_rasterio_shape()
                print("Clipping complete")
                dataset_dir = os.path.join(gi_dirs['RGB_OUTPUT_DIR'], 'Excel_' + rgb_dataset.filename)
                return jsonify({
                    'success': True,
                    'crs_matched': crs_matched,
                    'datasetDir': dataset_dir
                })
            elif crs_mismatch_message:
                print(f"CRS mismatch: {crs_mismatch_message}")
                return jsonify({'crs_mismatch': crs_mismatch_message}), 400
        return jsonify({'error': 'Missing required files'}), 400

    elif image_type == 'ms':
        print("Processing MS dataset")
        # Get files from MS directories
        band_paths = {
            'r_band': os.path.join(gi_dirs['MS_R_BAND_DIR'], os.listdir(gi_dirs['MS_R_BAND_DIR'])[0]),
            'g_band': os.path.join(gi_dirs['MS_G_BAND_DIR'], os.listdir(gi_dirs['MS_G_BAND_DIR'])[0]),
            'b_band': os.path.join(gi_dirs['MS_B_BAND_DIR'], os.listdir(gi_dirs['MS_B_BAND_DIR'])[0]),
            're_band': os.path.join(gi_dirs['MS_RE_BAND_DIR'], os.listdir(gi_dirs['MS_RE_BAND_DIR'])[0]),
            'nir_band': os.path.join(gi_dirs['MS_NIR_BAND_DIR'], os.listdir(gi_dirs['MS_NIR_BAND_DIR'])[0])
        }
        print("Found band files:", band_paths)
        
        shapefile_paths = [os.path.join(gi_dirs['MS_SHAPEFILE_DIR'], f) for f in os.listdir(gi_dirs['MS_SHAPEFILE_DIR'])]
        print(f"Found {len(shapefile_paths)} shapefile components")
        
        if all(os.path.exists(path) for path in band_paths.values()) and shapefile_paths:
            r_band_file = os.path.basename(band_paths['r_band'])
            filename = os.path.splitext(r_band_file)[0]
            dataset_dir = os.path.join(gi_dirs['MS_OUTPUT_DIR'], 'Excel_' + filename)
            print(f"Output directory: {dataset_dir}")
            
            global ms_dataset
            ms_dataset, crs_matched, crs_mismatch_message = create_ms_dataset(band_paths, shapefile_paths)
            
            if ms_dataset:
                print("MS dataset created successfully")
                ms_dataset.clip_rasterio_shape()
                print("Clipping complete")
                return jsonify({
                    'success': True,
                    'crs_matched': crs_matched,
                    'datasetDir': dataset_dir
                })
            elif crs_mismatch_message:
                print(f"CRS mismatch: {crs_mismatch_message}")
                return jsonify({'crs_mismatch': crs_mismatch_message}), 400
        return jsonify({'error': 'Missing required files'}), 400
        
    return jsonify({'error': 'Invalid image type'}), 400


def check_crs_match(gdf, src):
    if gdf.crs == src.crs:
        return True, None
    else:
        crs_mismatch_message = f'Shapefile CRS {gdf.crs}: {gdf.crs}\nImage CRS {src.crs}: {src.crs}\nThe shapefile CRS is different from the Image CRS. Do you want to reproject the shapefile to match the Image CRS?'
        return False, crs_mismatch_message

# DOWNLOAD DATASET
@geoimaging_bp.route('/download_dataset', methods=['GET'])
def download_dataset():
    dataset_dir = request.args.get('dir')
    dataset_type = request.args.get('type')
    
    print(f"Downloading dataset from: {dataset_dir}")
    print(f"Dataset type: {dataset_type}")
    
    if not dataset_dir or dataset_type not in ['rgb', 'ms']:
        return jsonify({'error': 'Missing directory or invalid type parameter'}), 400

    # Extract dataset
    if dataset_type == 'rgb':
        if rgb_dataset:
            output_dir = rgb_dataset.dataset_extraction_auto()
        else:
            return jsonify({'error': 'RGB dataset not initialized'}), 400
    elif dataset_type == 'ms':
        if ms_dataset:
            output_dir = ms_dataset.dataset_extraction_auto()
        else:
            return jsonify({'error': 'MS dataset not initialized'}), 400
    
    # Get the created Excel files
    excel_files = [f for f in os.listdir(output_dir) if f.endswith('.xlsx')]
    
    if not excel_files:
        return jsonify({'error': 'No dataset files found'}), 404
        
    dataset_file = excel_files[0]
    file_path = os.path.join(output_dir, dataset_file)
    
    try:
        return send_file(
            file_path,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=f'{dataset_type}_dataset_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
        )
    except Exception as e:
        print(f"Error sending file: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Reproject the shapefile to match the image CRS
@geoimaging_bp.route('/reproject_shapefile', methods=['POST'])
def reproject_shapefile():
    global crs_checked
    image_type = request.json.get('imageType')

    if image_type == 'rgb':
        global rgb_dataset
        if rgb_dataset is None:
            return jsonify({'success': False, 'error': 'RGB dataset not initialized'}), 400

        try:
            rgb_dataset.gdf = rgb_dataset.gdf.to_crs(rgb_dataset.src.crs)
            print(f"Shapefile has been reprojected to match the Image CRS.")
            crs_checked = True
            return jsonify({'success': True, 'message': 'Shapefile has been reprojected to match the Image CRS.'}), 200
        except Exception as e:
            print(f'Error reprojecting shapefile: {e}')
            return jsonify({'success': False, 'error': str(e)}), 500

    elif image_type == 'ms':
        global ms_dataset
        if ms_dataset is None:
            return jsonify({'success': False, 'error': 'MS dataset not initialized'}), 400

        try:
            ms_dataset.gdf = ms_dataset.gdf.to_crs(ms_dataset.src_r.crs)
            print(f"Shapefile has been reprojected to match the Image CRS.")
            crs_checked = True
            return jsonify({'success': True, 'message': 'Shapefile has been reprojected to match the Image CRS.'}), 200
        except Exception as e:
            print(f'Error reprojecting shapefile: {e}')
            return jsonify({'success': False, 'error': str(e)}), 500

    else:
        return jsonify({'success': False, 'error': 'Invalid image type'}), 400
#--------------------------------------------------------------------------------####

## UPLOAD IMAGES ##########################################################################################################
def upload_image(file, image_type):
    try:
        # Check if the file is provided
        if not file:
            return jsonify({'error': f'No {image_type} file provided'}), 400
        
        # Get directories from app config
        gi_dirs = current_app.config['GI_DIRS']

        # Determine the directory based on the image type
        if image_type == 'rgb':
            upload_dir = gi_dirs['RGB_IMAGE_DIR']
            filename = file.filename
        elif image_type == 'ms':
            band_type = request.form.get('band_type')
            if band_type not in ['r_band', 'g_band', 'b_band', 're_band', 'nir_band']:
                return jsonify({'error': 'Invalid band type'}), 400
            
            upload_dir = gi_dirs[f'MS_{band_type.upper()}_DIR']
            filename = f"{band_type}_{file.filename}"
        else:
            return jsonify({'error': 'Invalid image type'}), 400

        # Save the file to the corresponding directory
        file_path = os.path.join(upload_dir, filename)
        file.save(file_path)
        print(f'Saved {image_type} file: {filename}')
        print(f'File path: {file_path}')

        return jsonify({'success': True})

    except Exception as e:
        print(f'Error uploading {image_type} file: {str(e)}')
        return jsonify({'error': str(e)}), 500

# Route to upload RGB image
@geoimaging_bp.route('/upload_rgb_image', methods=['POST'])
def upload_rgb_image():
    rgb_image_file = request.files['file']
    return upload_image(rgb_image_file, 'rgb')

# Route to upload Multispectral band
@geoimaging_bp.route('/upload_ms_band', methods=['POST'])
def upload_ms_band():
    ms_band_file = request.files['file']
    return upload_image(ms_band_file, 'ms')
#--------------------------------------------------------------------------------####

## UPLOAD SHAPEFILES #######################################################################################################
@geoimaging_bp.route('/upload_rgb_shapefile', methods=['POST'])
def upload_rgb_shapefile():
    print('Received request to upload RGB shapefile')
    return upload_shapefile('rgb_shapefile', 'Plot_ID')

@geoimaging_bp.route('/upload_ms_shapefile', methods=['POST'])
def upload_ms_shapefile():
    print('Received request to upload MS shapefile')
    return upload_shapefile('ms_shapefile', 'Plot_ID')

def upload_shapefile(shapefile_field, plot_id_field):
    shapefile_files = request.files.getlist(shapefile_field)
    print(f'Received {shapefile_field} files:', shapefile_files)

    if shapefile_files:
        try:
            # Get directories from app config
            gi_dirs = current_app.config['GI_DIRS']
            
            # Determine correct directory
            upload_dir = gi_dirs['RGB_SHAPEFILE_DIR'] if shapefile_field == 'rgb_shapefile' else gi_dirs['MS_SHAPEFILE_DIR']
            
            shapefile_file_paths = []

            # Find the .shp file
            shp_file = next((file for file in shapefile_files if file.filename.endswith('.shp')), None)

            if shp_file:
                # Save the .shp file
                shp_file_path = os.path.join(upload_dir, shp_file.filename)
                shp_file.save(shp_file_path)
                shapefile_file_paths.append(shp_file_path)
                print(f'Saved file: {shp_file.filename}')

                # Save remaining files
                for file in shapefile_files:
                    if file != shp_file:
                        file_path = os.path.join(upload_dir, file.filename)
                        file.save(file_path)
                        shapefile_file_paths.append(file_path)
                        print(f'Saved file: {file.filename}')

                # Read and process shapefile
                gdf = gpd.read_file(shp_file_path)
                print('Read shapefile:', gdf)

                column_names = list(gdf.columns)
                polygon_ids = gdf[plot_id_field].unique().tolist()
                geometries = [geom.wkt for geom in gdf['geometry'].unique()]

                return jsonify({
                    'success': True, 
                    'column_names': column_names, 
                    'polygon_ids': polygon_ids, 
                    'geometries': geometries
                })
            else:
                return jsonify({'success': False, 'error': 'No .shp file found in the uploaded files'}), 400

        except Exception as e:
            print(f'Error processing {shapefile_field}:', e)
            return jsonify({'success': False, 'error': str(e)}), 500
    else:
        return jsonify({'success': False, 'error': f'No {shapefile_field} files provided'}), 400
#-----------------------------------------------------------------------------------------------------------------------------------------------------------

## CLIP IMAGES #################################################################################################################
@geoimaging_bp.route('/clip_rgb_image', methods=['POST'])
def clip_rgb_image():
    global rgb_dataset
    
    # First initialize the dataset
    image_files = os.listdir(current_app.config['GI_DIRS']['RGB_IMAGE_DIR'])
    shapefile_files = os.listdir(current_app.config['GI_DIRS']['RGB_SHAPEFILE_DIR'])
    
    if image_files and shapefile_files:
        image_path = os.path.join(current_app.config['GI_DIRS']['RGB_IMAGE_DIR'], image_files[0])
        shapefile_paths = [os.path.join(current_app.config['GI_DIRS']['RGB_SHAPEFILE_DIR'], f) for f in shapefile_files]
        
        rgb_dataset, crs_matched, crs_mismatch_message = create_rgb_dataset(image_path, shapefile_paths)
        
        if rgb_dataset:
            def clipping_task():
                with geoimaging_bp.app.app_context():
                    rgb_dataset.clip_rasterio_shape()
                    current_app.extensions['socketio'].emit('clipping_complete', {'type': 'rgb'}, namespace='/geoimaging')

            thread = threading.Thread(target=clipping_task)
            thread.start()

            return jsonify({
                'success': True,
                'message': 'Dataset initialized and clipping started',
                'crs_matched': crs_matched
            })
        elif crs_mismatch_message:
            return jsonify({'error': crs_mismatch_message}), 400
    
    return jsonify({'error': 'Missing required files'}), 400

@geoimaging_bp.route('/clip_ms_band', methods=['POST'])
def clip_ms_band():
    global ms_dataset
    
    # First initialize the dataset
    band_paths = {
        'r_band': os.path.join(current_app.config['GI_DIRS']['MS_R_BAND_DIR'], os.listdir(current_app.config['GI_DIRS']['MS_R_BAND_DIR'])[0]),
        'g_band': os.path.join(current_app.config['GI_DIRS']['MS_G_BAND_DIR'], os.listdir(current_app.config['GI_DIRS']['MS_G_BAND_DIR'])[0]),
        'b_band': os.path.join(current_app.config['GI_DIRS']['MS_B_BAND_DIR'], os.listdir(current_app.config['GI_DIRS']['MS_B_BAND_DIR'])[0]),
        're_band': os.path.join(current_app.config['GI_DIRS']['MS_RE_BAND_DIR'], os.listdir(current_app.config['GI_DIRS']['MS_RE_BAND_DIR'])[0]),
        'nir_band': os.path.join(current_app.config['GI_DIRS']['MS_NIR_BAND_DIR'], os.listdir(current_app.config['GI_DIRS']['MS_NIR_BAND_DIR'])[0])
    }
    
    shapefile_paths = [os.path.join(current_app.config['GI_DIRS']['MS_SHAPEFILE_DIR'], f) 
                      for f in os.listdir(current_app.config['GI_DIRS']['MS_SHAPEFILE_DIR'])]
    
    if all(os.path.exists(path) for path in band_paths.values()) and shapefile_paths:
        ms_dataset, crs_matched, crs_mismatch_message = create_ms_dataset(band_paths, shapefile_paths)
        
        if ms_dataset:
            def clipping_task():
                with geoimaging_bp.app.app_context():
                    ms_dataset.clip_rasterio_shape()
                    current_app.extensions['socketio'].emit('clipping_complete', {'type': 'ms'}, namespace='/geoimaging')

            thread = threading.Thread(target=clipping_task)
            thread.start()
            
            return jsonify({
                'success': True,
                'message': 'Dataset initialized and clipping started',
                'crs_matched': crs_matched
            })
        elif crs_mismatch_message:
            return jsonify({'error': crs_mismatch_message}), 400
    
    return jsonify({'error': 'Missing required files'}), 400

@geoimaging_bp.route('/get_clipped_images', methods=['GET'])
def get_clipped_images():
    image_type = request.args.get('type', 'rgb')
    gi_dirs = current_app.config['GI_DIRS']
    
    if image_type == 'rgb':
        if rgb_dataset is None:
            return jsonify({'error': 'RGB dataset not initialized'}), 400
        file_name = rgb_dataset.filename
        clip_dir = os.path.join(gi_dirs['RGB_OUTPUT_DIR'], 'shape_clip_' + file_name)
    else:
        if ms_dataset is None:
            return jsonify({'error': 'MS dataset not initialized'}), 400
        clip_dir = os.path.join(gi_dirs['MS_OUTPUT_DIR'], 'shape_clip')

    if not os.path.exists(clip_dir):
        return jsonify({'error': 'No clipped images found'}), 404
        
    images = []
    for filename in os.listdir(clip_dir):
        if filename.endswith('.tif'):
            plot_id = os.path.splitext(filename)[0]
            with rasterio.open(os.path.join(clip_dir, filename)) as src:
                img = src.read()
                img_bytes = io.BytesIO()
                plt.imsave(img_bytes, img.transpose(1,2,0), format='png')
                img_base64 = base64.b64encode(img_bytes.getvalue()).decode()
                images.append({
                    'plot_id': plot_id,
                    'data': img_base64
                })

    return jsonify({'images': images})

@geoimaging_bp.route('/check_files')
def check_files():
    rgb_image_files = os.listdir(current_app.config['GI_DIRS']['RGB_IMAGE_DIR'])
    rgb_shapefile_files = os.listdir(current_app.config['GI_DIRS']['RGB_SHAPEFILE_DIR'])
    return jsonify({
        'hasImage': len(rgb_image_files) > 0,
        'hasShapefile': len(rgb_shapefile_files) > 0
    })
#-----------------------------------------------------------------------------------------------------------------

## AUTO DATA EXTRACTION ########################################################################################################
@geoimaging_bp.route('/extract_auto', methods=['POST'])
def extract_auto():
    dataset_type = request.form.get('datasetType')

    if dataset_type == 'rgb':
        global rgb_dataset
        if rgb_dataset is None:
            return jsonify({'error': 'RGB dataset not initialized'}), 400

        rgb_dataset.dataset_extraction_auto()
        return jsonify({'success': True})

    elif dataset_type == 'ms':
        global ms_dataset
        if ms_dataset is None:
            return jsonify({'error': 'MS dataset not initialized'}), 400

        ms_dataset.dataset_extraction_auto()
        return jsonify({'success': True})

    else:
        return jsonify({'error': 'Invalid dataset type'}), 400
#---------------------------------------------------------------------------------------------------------------------

## MANUAL DATA EXTRACTION ########################################################################################################
@geoimaging_bp.route('/rgb_manual_extraction', methods=['GET'])
def rgb_manual_extraction():
    global rgb_dataset
    if rgb_dataset is None:
        return jsonify({'error': 'RGB dataset not initialized'}), 400

    initial_data = get_rgb_initial_data()
    return jsonify(initial_data)

def get_rgb_initial_data():
    vis = RGB2Dataset.calculate_vi(None, None, None, None, None)
    vis = [vi[0] for vi in vis]
    plot_data = rgb_dataset.dataset_extraction_manu_threshold(vis[0])
    return {'vis': vis, 'plotData': plot_data[:10]}

@geoimaging_bp.route('/get_rgb_plot_data', methods=['GET'])
def get_rgb_plot_data():
    global rgb_dataset
    if rgb_dataset is None:
        return jsonify({'error': 'RGB dataset not initialized'}), 400

    vi = request.args.get('vi')
    page = int(request.args.get('page', 1))
    plot_data = rgb_dataset.dataset_extraction_manu_threshold(vi)
    start_index = (page - 1) * 10
    end_index = start_index + 10
    page_data = plot_data[start_index:end_index]
    return jsonify({'plotData': page_data, 'currentPage': page})

@geoimaging_bp.route('/get_rgb_histogram_data', methods=['POST'])
def get_rgb_histogram_data():
    global rgb_dataset
    if rgb_dataset is None:
        return jsonify({'error': 'RGB dataset not initialized'}), 400

    vi = request.form.get('vi')
    plot_index = int(request.form.get('plot_index'))
    hist_or_cc = request.form.get('hist_or_cc') == 'true'

    data_for_window = rgb_dataset.dataset_extraction_manu_threshold(vi)

    if data_for_window:
        plot_data = data_for_window[plot_index]
        threshold = plot_data[2]

        if hist_or_cc:
            img_bytes = rgb_dataset.dataset_extraction_manu_plot(vi, threshold, plot_index, True)
        else:
            img_bytes = rgb_dataset.dataset_extraction_manu_plot(vi, threshold, plot_index, False)

        img_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
        return jsonify({'image_data': img_base64})
    else:
        return jsonify({'error': 'Failed to fetch vegetation index data'}), 400

@geoimaging_bp.route('/rgb_manual_extraction_ok', methods=['POST'])
def rgb_manual_extraction_ok():
    global rgb_dataset
    if rgb_dataset is None:
        return jsonify({'error': 'RGB dataset not initialized'}), 400

    vi = request.form.get('vi')
    thresholds = request.form.getlist('thresholds[]')
    rgb_dataset.dataset_extraction_manu_ok(vi, thresholds)
    return jsonify({'message': 'Manual extraction completed successfully'})

## MS ##
# Route to fetch initial data for MS manual extraction
@geoimaging_bp.route('/ms_manual_extraction', methods=['GET'])
def ms_manual_extraction():
    global ms_dataset
    if ms_dataset is None:
        return jsonify({'error': 'MS dataset not initialized'}), 400

    initial_data = get_ms_initial_data()
    return jsonify(initial_data)

def get_ms_initial_data():
    bands = ['red', 'green', 'blue', 'rededge', 'nir']
    plot_data = ms_dataset.dataset_extraction_manu_threshold('NDVI')
    return {'bands': bands, 'plotData': plot_data[:10]}

# Route to fetch plot data for the selected band and page
@geoimaging_bp.route('/get_ms_plot_data', methods=['GET'])
def get_ms_plot_data():
    global ms_dataset
    if ms_dataset is None:
        return jsonify({'error': 'MS dataset not initialized'}), 400

    band = request.args.get('band')
    page = int(request.args.get('page', 1))
    plot_data = ms_dataset.dataset_extraction_manu_threshold(band)
    start_index = (page - 1) * 10
    end_index = start_index + 10
    page_data = plot_data[start_index:end_index]
    return jsonify({'plotData': page_data, 'currentPage': page})

# Route to handle histogram and color composite requests
@geoimaging_bp.route('/get_ms_histogram_data', methods=['POST'])
def get_ms_histogram_data():
    global ms_dataset
    if ms_dataset is None:
        return jsonify({'error': 'MS dataset not initialized'}), 400

    band = request.form.get('band')
    plot_index = int(request.form.get('plot_index'))
    hist_or_cc = request.form.get('hist_or_cc') == 'true'
    selected_band_show = request.form.get('selected_band_show')

    data_for_window = ms_dataset.dataset_extraction_manu_threshold(band)

    if data_for_window:
        plot_data = data_for_window[plot_index]
        threshold = plot_data[2]

        if hist_or_cc:
            img_bytes = ms_dataset.dataset_extraction_manu_plot(band, threshold, plot_index, True, selected_band_show)
        else:
            img_bytes = ms_dataset.dataset_extraction_manu_plot(band, threshold, plot_index, False, selected_band_show)

        img_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
        return jsonify({'image_data': img_base64})
    else:
        return jsonify({'error': 'Failed to fetch band data'}), 400

# Route to handle "OK" button click for MS manual extraction
@geoimaging_bp.route('/ms_manual_extraction_ok', methods=['POST'])
def ms_manual_extraction_ok():
    global ms_dataset
    if ms_dataset is None:
        return jsonify({'error': 'MS dataset not initialized'}), 400

    band = request.form.get('band')
    thresholds = request.form.getlist('thresholds[]')
    ms_dataset.dataset_extraction_manu_ok(band, thresholds)
    return jsonify({'message': 'Manual extraction completed successfully'})
#--------------------------------------------------------------------#

## HISTOGRAM / CC / SHAPEFILE VISUALIZATION #############################################
@geoimaging_bp.route('/visualize_shapefile', methods=['POST'])
def visualize_shapefile():
    data_type = request.json.get('dataType')

    if data_type == 'rgb':
        global rgb_dataset

        if rgb_dataset is None:
            return jsonify({'error': 'RGB dataset not initialized'}), 400

        try:
            img_bytes = rgb_dataset.check_original()
            img_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
            return jsonify({'message': 'Shapefile visualization successful', 'image_data': img_base64})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    elif data_type == 'ms':
        global ms_dataset

        if ms_dataset is None:
            return jsonify({'error': 'MS dataset not initialized'}), 400

        try:
            selected_band_show = request.json.get('selectedBandShow', 'nir')
            img_bytes = ms_dataset.gdf_visualization(selected_band_show)
            img_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
            return jsonify({'message': 'Shapefile visualization successful', 'image_data': img_base64})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    else:
        return jsonify({'error': 'Invalid data type'}), 400

@geoimaging_bp.route('/get_max_index_range', methods=['POST'])
def get_max_index_range():
    data_type = request.json.get('dataType')

    if data_type == 'rgb':
        global rgb_dataset
        if rgb_dataset is None:
            return jsonify({'error': 'RGB dataset not initialized'}), 400
        max_index_range = len(rgb_dataset.gdf) - 1
    elif data_type == 'ms':
        global ms_dataset
        if ms_dataset is None:
            return jsonify({'error': 'MS dataset not initialized'}), 400
        max_index_range = len(ms_dataset.gdf) - 1
    else:
        return jsonify({'error': 'Invalid data type'}), 400

    return jsonify({'max_index_range': max_index_range})

@geoimaging_bp.route('/handle_plot_index', methods=['POST'])
def handle_plot_index():
    data_type = request.json.get('dataType')
    plot_index = request.json.get('plot_index')
    is_histogram = request.json.get('is_histogram')
    selected_band_show = request.json.get('selectedBandShow', 'nir')

    # Create a Queue object to receive the result from the thread
    result_queue = Queue()

    # Create a new thread to generate the Matplotlib plot
    def generate_plot_thread():
        if data_type == 'rgb':
            global rgb_dataset
            if rgb_dataset is not None:
                img_bytes = rgb_dataset.visualization_plot(plot_index, is_histogram)
                result_queue.put(img_bytes)

        elif data_type == 'ms':
            global ms_dataset
            if ms_dataset is not None:
                img_bytes = ms_dataset.visualization_plot(plot_index, is_histogram, selected_band_show)
                result_queue.put(img_bytes)

        else:
            result_queue.put(None)
    
    if data_type == 'rgb':
        global rgb_dataset

        if rgb_dataset is None:
            return jsonify({'error': 'RGB dataset not initialized'}), 400

        if plot_index is not None:
            # Get the index range based on the length of the geodataframe
            index_range = len(rgb_dataset.gdf) - 1

            if plot_index < 0 or plot_index > index_range:
                return jsonify({'error': 'Invalid plot index'}), 400

            # Do something with the selected plot index
            print(f'Selected Plot Index: {plot_index}')
        else:
            return jsonify({'error': 'Invalid plot index'}), 400

    elif data_type == 'ms':
        global ms_dataset

        if ms_dataset is None:
            return jsonify({'error': 'MS dataset not initialized'}), 400

        if plot_index is not None:
            # Get the index range based on the length of the geodataframe
            index_range = len(ms_dataset.gdf) - 1

            if plot_index < 0 or plot_index > index_range:
                return jsonify({'error': 'Invalid plot index'}), 400

            # Do something with the selected plot index
            print(f'Selected Plot Index: {plot_index}')
        else:
            return jsonify({'error': 'Invalid plot index'}), 400
    # Start the thread and wait for the result
    thread = threading.Thread(target=generate_plot_thread)
    thread.start()

    # Wait for the result from the thread
    img_bytes = result_queue.get()

    # Return the image data if it was generated successfully
    if img_bytes:
        img_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
        return jsonify({'image_data': img_base64})

    # Handle other cases or return an error if necessary
    return jsonify({'error': 'Invalid data type or dataset not initialized'}), 400

@geoimaging_bp.route('/plot_index_popup')
def plot_index_popup():
    return render_template('plot_index_popup.html')

@geoimaging_bp.route('/band_selection_popup')
def band_selection_popup():
    return render_template('band_selection_popup.html')
#---------------------------------------------------------------------#

# UTILITIES FUNCTIONS #############################################################################
def otsu(data, is_plot):
    """
    Input: 
        data: vegetation index data. 
        is_plot: if it is true, will return hist,bins,and optimal_threshold; otherwise, only need to return optimal_threshold.
    
    """
    hist,bins = np.histogram(data,bins= 100,range=(np.min(data),np.max(data)))
    # Calculate the histogram of the input data using 100 bins
    # The range is set from the minimum to the maximum value of the input data

    ## ostu method
    total_freq = sum(hist)
    # Calculate the total frequency by summing the histogram values
    
    # Compute probabilities of each intensity level
    probabilities =  hist/ total_freq
    # Compute the probabilities by dividing the histogram values by the total frequency

    # Initialize variables for Ostu's method
    max_variance = 0
    optimal_threshold = 0

    for t in range(1, len(bins)):
        # Iterate over the possible thresholds (excluding the first bin)
        w0 = np.sum(probabilities [:t])
        w1 = np.sum(probabilities [t:])
        # Calculate the weights (probabilities) for the two classes (0 and 1) separated by the current threshold

        if 0 == w0:
            break
        # If the weight for class 0 is 0, skip this iteration (no need to calculate further)

        if 0 == w1:
            continue
        # If the weight for class 1 is 0, continue to the next iteration
        
        mean0 = np.sum(probabilities [:t] * bins[:t])/w0
        mean1 = np.sum(probabilities [t:] * bins[t:-1])/w1
        # Calculate the mean values for the two classes

        variance = w0 *w1 *(mean0-mean1) **2
        # Calculate the variance between the two classes

        if variance >max_variance:
            max_variance = variance
            optimal_threshold = bins[t]
        # Update the maximum variance and corresponding optimal threshold

    if is_plot ==True:
        return hist,bins,optimal_threshold
    else:
        return optimal_threshold
    

def plot_VI_hist(ax,title,data,plot_index):
    """
    This function plots the histogram of the given vegetation index data.
    
    Args:
        ax (matplotlib.axes.Axes): The axes object on which to plot the histogram.
        title (str): The title of the vegetation index.
        data (numpy.ndarray): The vegetation index data.
        plot_index (str): The plot index or name.
    """
    hist,bins,optimal_threshold= otsu(data,True)
    # Call the otsu() function to get the histogram, bins, and optimal threshold for the given data

    # plot the hist of VI
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i/len(hist)) for i in range(len(hist))]
    # Get a colormap and create a list of colors based on the histogram length

    if title == 'ExR' or title =='CIVE':
        colors = colors[::-1]
    # Reverse the colors list if the title is 'ExR' or 'CIVE'

    ax.bar(bins[:-1], hist,width=np.diff(bins),color = colors, label ='Color')
    # Plot the histogram bars with the calculated colors

    sm = ScalarMappable(cmap='viridis')
    sm.set_array([data])
    cbar = plt.colorbar(sm, ax = ax,orientation='vertical',shrink=0.3)
    cbar.set_label('Vegetation Index Value')
    # Create a colorbar and set its label

    # Anzeigen der Werte in einer Box
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                            edgecolor='none', linewidth=0)
    name_title = f'Plot{plot_index}_{title}' +'\n'+ f'Th= {optimal_threshold:.2f} '
    ax.legend([extra], [name_title], loc='upper left')
    # Create a legend with the plot index, title, and optimal threshold

    # ax.set_title()
    ax.set_xlabel('Vegetation Index Value')
    ax.set_ylabel('Frequency')
    # Set the x and y labels for the histogram plot

###########################RGB#######################################
class RGB2Dataset():
    def __init__(self,src,gdf,filename,output_dir,clip_dir):
        self.src = src
        self.gdf = gdf
        self.filename = filename
        self.output_dir = current_app.config['GI_DIRS']['RGB_OUTPUT_DIR']
        self.clip_dir = os.path.join(output_dir, 'shape_clip_' + filename)
        
    @staticmethod
    def calculate_canpoy_cover(input_dir):
        """
        Input parameters:input_dir is a dictionary(key-value pair)
            input_dir: 'vi', VI values
                    'threshold': calculated threshold from ostu
                    'title': vi name
                    'red':red band pixel values
                    'green': green band pixel values
                    'blue':blue band pixel values
        return: 
            'cc':vegetation_cover_ratio
            'red':red_mean
            'green':green_mean
            'blue':blue_mean         
        """
        
        VI_selected = input_dir['vi']
        selected_threshold = input_dir['threshold']
        title = input_dir['title']
        red = input_dir['red']
        green = input_dir['green']
        blue = input_dir['blue']

        # Apply thresholds to create a mask
        if title == 'ExR' or title == "CIVE":
            mask =  VI_selected >= selected_threshold
        else:     
            mask =  VI_selected < selected_threshold 

        VI_masked = np.where(mask,np.nan,VI_selected) # mask the non veg pixels

        vegetation_pixels = np.sum(~np.isnan(VI_masked)) # extrat the veg pixels

        total_pixels = np.prod(VI_masked.shape)-np.sum(np.isnan(VI_selected)) # the total pixels

        vegetation_cover_ratio = vegetation_pixels / total_pixels  # canopy cover

        red_masked = np.where(mask,np.nan,red)
        green_masked = np.where(mask,np.nan,green)
        blue_masked = np.where(mask,np.nan,blue)

        red_masked = red_masked[~np.isnan(red_masked)]
        green_masked = green_masked[~np.isnan(green_masked)]
        blue_masked = blue_masked[~np.isnan(blue_masked)]

        red_mean = np.mean(red_masked)
        green_mean = np.mean(green_masked)
        blue_mean =np.mean(blue_masked)

        return {'cc':vegetation_cover_ratio,'red':red_mean, 'green':green_mean,'blue':blue_mean}

    @staticmethod
    def calculate_vi(red,green,blue,selected_vi):

        if selected_vi == None:

            exg = 2*green - red - blue
            exr = 1.3*red -green
            ndi = 128*(((green-red)/(green +red))+1)
            cive = 0.441*red-0.811*green+0.385*blue +18.78745 
            exgr = exg -exr
            veg = green/(red**0.667 * blue**(1-0.667))
            com1 = exg + cive + exgr + veg
            mexg = 1.262 *green -0.884*red -0.311*blue
            com2 = 0.36 *exg + 0.47*cive + 0.17*veg

            VIs = [('NDI' , ndi),('ExG' , exg), ('ExR' , exr), ('CIVE' , cive), ('ExGR' , exgr), ('COM1', com1), ('MExG' , mexg), ('COM2' , com2),('VEG' , veg)]
            
            return  VIs
        else:

            match selected_vi:
                case 'NDI':
                    vi = 128*(((green-red)/(green +red))+1)
                case 'ExG':
                    vi = 2*green - red - blue
                case 'ExR':
                    vi = 1.3*red -green
                case 'CIVE':
                    vi =0.441*red-0.811*green+0.385*blue +18.78745
                case 'ExGR':
                    exg = 2*green - red - blue
                    exr = 1.3*red -green
                    vi = exg -exr
                case 'COM1':
                    exg = 2*green - red - blue
                    exr = 1.3*red -green
                    cive = 0.441*red-0.811*green+0.385*blue +18.78745
                    exgr = exg -exr
                    veg = green/(red**0.667 * blue**(1-0.667))
                    vi= exg + cive + exgr + veg 
                case 'MExG':
                    vi =1.262 *green -0.884*red -0.311*blue
                case 'COM2':
                    vi = 0.36 *exg + 0.47*cive + 0.17*veg
                case 'VEG':
                    vi = green/(red**0.667 * blue**(1-0.667))
            return vi        
            
    ## function 1: Clip image.
    # Clipping raw data based on the shape file boundry and store them in desired locations
    
    def clip_rasterio_shape(self):
        """
            Input: 
                image_path: path to image
                shapefile_path: path to your shapefile
        """
        # Create output directory using proper path joining
        output_path = os.path.join(self.output_dir, 'shape_clip_' + self.filename)
        
        # Clean up existing directory if it exists
        if os.path.exists(output_path):
            try:
                shutil.rmtree(output_path)
                time.sleep(0.5)  # Give OS time to complete deletion
            except Exception as e:
                print(f"Warning: Could not remove existing directory: {e}")
        
        # Create fresh directory
        os.makedirs(output_path, exist_ok=True)

        # Clip image to the shapefile geometry
        clipped_image, clipped_transform = mask(self.src, self.gdf.geometry, crop=True)

        # Specify output path for each individual band using the image name
        clipped_image_path = os.path.join(output_path, f'{self.filename}-clipped.tif')

        # Create a new raster file with the same dimensions as the clipped band
        with rasterio.open(clipped_image_path, 'w', driver='GTiff', width=clipped_image.shape[2], height=clipped_image.shape[1], count=4, dtype=clipped_image.dtype, crs=self.src.crs, transform=clipped_transform) as dst:
            dst.write(clipped_image)
        print(f'clipped image {self.filename}-clipped.tif saved to {clipped_image_path}')
        
        # Print shape and CRS (sanity check)
        print(f'Shape: {clipped_image.shape}, CRS: {self.src.crs}')
        
        # loop through the selected polygons and save each clipped image separately
        for polygon_idx, polygon in enumerate(self.gdf.geometry):
            # Extract the geometry of the selected polygon
            polygon_geometry = self.gdf.iloc[polygon_idx]['geometry']
            plot_name = self.gdf.iloc[polygon_idx]['Plot_ID']   # here we cannot fix it, should be changable.

            # Clip the image to the selected polygon's geometry and specify the output CRS
            clipped_image, clipped_transform = mask(self.src, [polygon_geometry], crop=True)
            clipped_meta = self.src.meta.copy()
            clipped_meta['crs'] = {'init': f'epsg:{self.gdf.crs}'}

            # output file path for the clipped image using the band and location index
            output_image_path = os.path.join(output_path, f'{plot_name}.tif')

            # Create a new raster file with the same dimensions as the clipped image
            with rasterio.open(output_image_path, 'w', driver='GTiff', width=clipped_image.shape[2], height=clipped_image.shape[1], count=clipped_image.shape[0], dtype=clipped_image.dtype, crs=self.src.crs, transform=clipped_transform) as dst:
                dst.write(clipped_image)

        print("Clipped images saved in the output directories for each plot.")
    
    def dataset_extraction_auto(self):
        print("\n=== Starting Auto Dataset Extraction ===")
        
        # 1. Create output directory using proper path joining
        auto_extract_dir = os.path.join(self.output_dir, 'Excel_' + self.filename)
        os.makedirs(auto_extract_dir, exist_ok=True)
        print(f"Created output directory: {auto_extract_dir}")

        # 2. Initialize arrays for storing results
        plot_num = self.gdf.shape[0]
        cc_all = [[None] * 9 for _ in range(plot_num)]
        red_all = [[None] * 9 for _ in range(plot_num)]
        green_all = [[None] * 9 for _ in range(plot_num)]
        blue_all = [[None] * 9 for _ in range(plot_num)]

        # 3. Process each plot using pre-clipped images
        for polygon_idx, (plot_id, geometry) in enumerate(zip(self.gdf['Plot_ID'], self.gdf.geometry)):
            print(f"\nProcessing plot {polygon_idx + 1}/{plot_num}: {plot_id}")
            
            # Load pre-clipped image
            clipped_src = rasterio.open(os.path.join(self.clip_dir, f'{plot_id}.tif'))
            clipped_image, _ = mask(clipped_src, [geometry], crop=True)
            
            # Process image data
            masked_img = np.where(clipped_image <= 0, np.nan, clipped_image)
            red = masked_img[0,:,:]
            green = masked_img[1,:,:]
            blue = masked_img[2,:,:]

            # Calculate VIs and metrics
            VIs = RGB2Dataset.calculate_vi(red, green, blue, None)
            cc = []
            red_mean = []
            green_mean = []
            blue_mean = []

            for (title, vi) in VIs:
                vi_values = vi.flatten()
                vi_values = vi_values[~np.isnan(vi_values)]
                threshold = otsu(vi_values, False)
                
                var_tuple = {'title': title, 'vi': vi, 'threshold': threshold,
                            'red': red, 'green': green, 'blue': blue}
                output = RGB2Dataset.calculate_canpoy_cover(var_tuple)
                
                cc.append(output['cc'])
                red_mean.append(output['red'])
                green_mean.append(output['green'])
                blue_mean.append(output['blue'])

            # Store results
            cc_all[polygon_idx] = cc
            red_all[polygon_idx] = red_mean
            green_all[polygon_idx] = green_mean
            blue_all[polygon_idx] = blue_mean

        # 4. Create initial DataFrame with all VIs
        col_name = ['NDI', 'ExG', 'ExR', 'CIVE', 'ExGR', 'COM1', 'MExG', 'COM2', 'VEG']
        df = pd.DataFrame(cc_all, columns=col_name)
        df.insert(0, column='Plot_ID', value=self.gdf['Plot_ID'])
        print("\nCreated initial DataFrame with all VIs")

        # 5. Calculate correlations between VIs
        correlations = {}
        for col1 in df.columns[1:]:
            for col2 in df.columns[1:]:
                if col1 < col2:  # Avoid duplicate comparisons
                    correlation = df[col1].corr(df[col2])
                    correlations[f"{col1}-{col2}"] = correlation
        
        # Find highest correlating VIs
        max_corr_value = max(correlations.values())
        max_corr_pairs = [pair for pair, corr in correlations.items() 
                        if corr == max_corr_value]
        print(f"\nFound {len(max_corr_pairs)} VI pairs with correlation {max_corr_value}")

        # 6. Create and save final datasets
        for pair in max_corr_pairs:
            vi1, vi2 = pair.split('-')
            df_final = pd.DataFrame({
                'Plot_ID': self.gdf['Plot_ID'],
                f'CC_{vi1}': df[vi1],
                f'CC_{vi2}': df[vi2],
                'Red': [row[df.columns.get_loc(vi1)-1] for row in red_all],
                'Green': [row[df.columns.get_loc(vi1)-1] for row in green_all],
                'Blue': [row[df.columns.get_loc(vi1)-1] for row in blue_all]
            })

            # Add all VIs to final DataFrame
            for title, _ in RGB2Dataset.calculate_vi(
                df_final['Red'].values,
                df_final['Green'].values,
                df_final['Blue'].values,
                None
            ):
                df_final[title] = _

            # Save to Excel
            excel_filepath = os.path.join(
                auto_extract_dir, 
                f'dataset_{vi1}_{vi2}_{max_corr_value:.2f}_{self.filename}.xlsx'
            )
            df_final.to_excel(excel_filepath, index=False)
            print(f"Saved dataset to: {excel_filepath}")

        print('\n=== Auto Dataset Extraction Complete ===')
        return auto_extract_dir

    # Test: select a plot to see
    def visualization_plot(self, plot_index, hist_or_cc):
        """
        This method visualizes either the histogram of vegetation index (VI) values or the vegetation pixels for a selected plot.

        Args:
            self: The instance of the RGB2Dataset class.
            plot_index (int): The index of the plot to visualize.
            hist_or_cc (bool): If True, visualize the histogram of VI values; if False, visualize the vegetation pixels.
        """
        # Create a BytesIO object to save the plot image
        img = io.BytesIO()
        # Extract the geometry of the selected polygon
        polygon_geometry = self.gdf.iloc[plot_index]['geometry']
        plot_name = self.gdf.iloc[plot_index]['Plot_ID']
        # Get the geometry and plot ID of the selected plot from the shapefile

        # Clip the image to the selected polygon's geometry and specify the output CRS
        clipped_image, clipped_transform = mask(self.src, [polygon_geometry], crop=True)
        # Clip the input image to the selected plot's geometry using rasterio.mask.mask()

        # Mask black background
        mask_black = (clipped_image <= 0)
        masked_img = np.where(mask_black, np.nan, clipped_image)
        # Set pixel values <= 0 (black background) to NaN

        red = masked_img[0, :, :]
        green = masked_img[1, :, :]
        blue = masked_img[2, :, :]
        # Extract the red, green, and blue band values from the masked image

        # Calculate the VI values
        VIs = RGB2Dataset.calculate_vi(red, green, blue, None)
        # Calculate all vegetation index (VI) values using the calculate_vi() method

        fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(12, 8))
        axs = np.ravel(axs)
        # Create a 3x3 grid of subplots

        if hist_or_cc:
            # If hist_or_cc is True, visualize the histogram of VI values
            for ax, (title, vi) in zip(axs, VIs):
                vi_values = vi.flatten()  # Flatten the image array to a 1D array
                vi_values = vi_values[~np.isnan(vi_values)]  # Keep only the non-NaN VI values
                plot_VI_hist(ax, title, vi_values, plot_name)  # Plot the histogram of VI values

            plt.suptitle(f'{plot_name}:vegetation index hist')
            plt.tight_layout()
            img.seek(0)  # Reset the BytesIO object's position to the beginning
            # Save the plot to the BytesIO object instead of showing it
            plt.savefig(img, format='png')

        else:
            # If hist_or_cc is False, visualize the vegetation pixels
            threshold_all = []

            for (title, vi) in VIs:
                vi_values = vi.flatten()  # Flatten the image array to a 1D array
                vi_values = vi_values[~np.isnan(vi_values)]  # Keep only the non-NaN VI values
                threshold = otsu(vi_values, False)  # Calculate the threshold using the Otsu method
                threshold_all.append(threshold)

            VIs_th = [(v[0], v[1], t) for v, t in zip(VIs, threshold_all)]
            # Create a list of tuples containing the VI name, VI values, and threshold

            for ax, (title, vi, threshold) in zip(axs, VIs_th):
                VI_selected = vi
                threshold_selected = threshold

                # Apply thresholds to create a mask
                mask_selected = VI_selected < threshold_selected

                if title == 'ExR' or title == "CIVE":
                    mask_selected = VI_selected >= threshold_selected
                # For ExR and CIVE, the mask is created using the opposite condition

                VI_masked = np.where(mask_selected, np.nan, VI_selected)  # Mask the non-vegetation pixels
                vegetation_pixels = np.sum(~np.isnan(VI_masked))  # Count the number of vegetation pixels
                total_pixels = np.prod(VI_masked.shape) - np.sum(np.isnan(VI_selected))  # Count the total number of pixels
                vegetation_cover_ratio = vegetation_pixels / total_pixels  # Calculate the vegetation cover ratio

                # Plot vegetation pixels on the original image
                pure_soil_pixel_coords = [(col, row) for row, col in zip(*np.where(~np.isnan(VI_masked)))]
                crs_coords = [(col, row) for col, row in pure_soil_pixel_coords]  # Get the CRS coordinates of the vegetation pixels

                # Plot the scatter plot for the vegetation pixel coordinates in CRS coordinates
                crs_x_coords, crs_y_coords = zip(*crs_coords)
                ax.scatter(crs_x_coords, crs_y_coords, c='red', marker='x', s=1, label='Pure veg Pixels')

                # # Display image and Set the labels and colorbar for the image
                # extent = (self.src.bounds.left, self.src.bounds.right, self.src.bounds.bottom, self.src.bounds.top)
                ax.imshow(clipped_image.transpose(1, 2, 0), cmap='RdYlGn', aspect=1)
                ax.axis('off')
                ax.set_title(f'{title} : CC={vegetation_cover_ratio}')

            # axs[-1].imshow(clipped_image.transpose(1,2,0), cmap='RdYlGn', aspect=1)
            # axs[-1].axis('off')
            plt.suptitle(f'{plot_name}:vegetation pixels')
            plt.tight_layout()
            img.seek(0)  # Reset the BytesIO object's position to the beginning
            # Save the plot to the BytesIO object instead of showing it
            plt.savefig(img, format='png')

        # Return the BytesIO object
        return img

    def visualization_shpfile(self):
        """
        This method visualizes the shapefile by plotting the geometries and annotating the plot IDs.

        Args:
            self: The instance of the RGB2Dataset class.
        """
        # Plot the geometries from the shapefile
        ax = self.gdf.geometry.plot(figsize=(12, 8))
        # Create a plot of the geometries from the shapefile (self.gdf) with a figure size of 12x8 inches

        # Annotate the plot IDs on the plot
        self.gdf.apply(lambda x: ax.annotate(text=x['Plot_ID'], xy=x.geometry.centroid.coords[0], ha='center'), axis=1)
        # For each row in the shapefile (self.gdf), annotate the plot ID (x['Plot_ID']) at the centroid of the geometry (x.geometry.centroid.coords[0])
        # The 'ha' parameter specifies the horizontal alignment of the annotation text, which is set to 'center'
        # The 'axis=1' parameter applies the lambda function to each row of the shapefile

    def check_original(self):
        """
        This method performs a sanity check for proper clipping by merging the shapefile on top of the stacked images.

        Args:
            self: The instance of the RGB2Dataset class.
        """
        # Sanity check for proper clipping (by merging shape file on the top of the stacked images)

        # Create a BytesIO object to save the plot image
        img = io.BytesIO()
        # Create a GeoDataFrame with the clipped geometry
        gdf_clipped = gpd.GeoDataFrame({'geometry': [g for g in self.gdf.geometry]}, crs=self.gdf.crs)
        # Create a new GeoDataFrame 'gdf_clipped' with the geometries from the original shapefile (self.gdf) and the same CRS

        # Plot each band image with the CRS coordinates and their band names
        image = self.src.read()
        # Read the image data from the input raster file (self.src)

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        # Create a new figure and axis with a size of 12x8 inches

        im = rasterio.plot.show(image, transform=self.src.transform, ax=ax, cmap='viridis')  # You can specify a colormap (e.g., 'viridis')
        # Plot the image data on the axis using rasterio.plot.show() with the specified colormap ('viridis')

        gdf_clipped.boundary.plot(ax=ax, color='red')  # Add the shapefile boundaries for reference
        # Plot the boundaries of the clipped geometries from 'gdf_clipped' on the axis in red color

        self.gdf.apply(lambda x: ax.annotate(text=x['Plot_ID'], xy=x.geometry.centroid.coords[0], ha='center'), axis=1)
        # Annotate the plot IDs from the original shapefile (self.gdf) on the axis at the centroid of each geometry

        ax.set_title(f"Band {self.filename} in CRS Coordinates")
        # Set the title of the plot with the filename of the input raster file

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        # Set the labels for the x and y axes as "Longitude" and "Latitude", respectively

        plt.tight_layout()
        # Adjust the layout of the plot to ensure all elements are visible
        img.seek(0)  # Reset the BytesIO object's position to the beginning
        # Save the plot to the BytesIO object instead of showing it
        plt.savefig(img, format='png')

        # Return the BytesIO object
        return img
    
    def check_clipped(self):
        """
        This method plots the clipped image with the shapefile boundaries and plot IDs.

        Args:
            self: The instance of the RGB2Dataset class.
        """
        # src = rasterio.open(clipped_image_path)
        # Commented out line to open a clipped image file

        img = io.BytesIO()

        clipped_image, clipped_transform = mask(self.src, self.gdf.geometry, crop=True)
        # Clip the input image (self.src) to the geometries in the shapefile (self.gdf.geometry) using rasterio.mask.mask()
        # The 'crop=True' parameter crops the output raster to the extent of the input geometries

        # Create a GeoDataFrame with the clipped geometry
        gdf_clipped = gpd.GeoDataFrame({'geometry': [g for g in self.gdf.geometry]}, crs=self.gdf.crs)
        # Create a new GeoDataFrame 'gdf_clipped' with the geometries from the original shapefile (self.gdf) and the same CRS

        # Plot each band image with the CRS coordinates and their band names
        image = clipped_image
        # Use the clipped image data for plotting

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        # Create a new figure and axis with a size of 10x10 inches

        # mask_black = image <= 0
        # image = np.where(mask_black, np.nan, image)
        # Commented out lines to mask black background pixels

        im = rasterio.plot.show(image, transform=clipped_transform, ax=ax, cmap='viridis')  # You can specify a colormap (e.g., 'viridis')
        # Plot the clipped image data on the axis using rasterio.plot.show() with the specified colormap ('viridis') and the clipped transform

        # ax.imshow(image.transpose(1, 2, 0), cmap='viridis')
        # Commented out line to plot the image using ax.imshow()

        gdf_clipped.boundary.plot(ax=ax, color='red')  # Add the shapefile boundaries for reference
        # Plot the boundaries of the clipped geometries from 'gdf_clipped' on the axis in red color

        self.gdf.apply(lambda x: ax.annotate(text=x['Plot_ID'], xy=x.geometry.centroid.coords[0], ha='center', color='red'), axis=1)
        # Annotate the plot IDs from the original shapefile (self.gdf) on the axis at the centroid of each geometry in red color

        ax.set_title(f"{self.filename} in CRS Coordinates")
        # Set the title of the plot with the filename of the input raster file

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        # Set the labels for the x and y axes as "Longitude" and "Latitude", respectively

        plt.tight_layout()
        # Adjust the layout of the plot to ensure all elements are visible
        img.seek(0)  # Reset the BytesIO object's position to the beginning
        # Save the plot to the BytesIO object instead of showing it
        plt.savefig(img, format='png')

        # Return the BytesIO object
        return img

    def dataset_extraction_manu_threshold(self, selected_vi):
        """
        This method calculates the vegetation index (VI) values, thresholds, and other statistics for each plot in the shapefile
        for a selected VI. The results are stored in a list of lists, where each inner list contains the plot index, plot ID,
        Otsu threshold, minimum VI value, and maximum VI value.

        Args:
            self: The instance of the RGB2Dataset class.
            selected_vi (str): The name of the selected vegetation index.

        Returns:
            data_for_window (list): A list of lists, where each inner list contains the plot index, plot ID, Otsu threshold,
                                    minimum VI value, and maximum VI value for a plot.
        """
        data_for_window = []
        # Initialize an empty list to store the results

        # Loop through the selected polygons and save each clipped image separately
        for polygon_idx, polygon in enumerate(self.gdf.geometry):
            # Extract the geometry of the selected polygon
            polygon_geometry = self.gdf.iloc[polygon_idx]['geometry']
            plot_name = self.gdf.iloc[polygon_idx]['Plot_ID']
            # Get the geometry and plot ID of the current plot from the shapefile

            # Clip the image to the selected polygon's geometry and specify the output CRS
            clipped_image, clipped_transform = mask(self.src, [polygon_geometry], crop=True)
            # Clip the input image to the current plot's geometry using rasterio.mask.mask()

            # Clear out the background fill
            mask_black = (clipped_image <= 0)
            masked_img = np.where(mask_black, np.nan, clipped_image)
            # Set pixel values <= 0 (black background) to NaN

            red = masked_img[0, :, :]
            green = masked_img[1, :, :]
            blue = masked_img[2, :, :]
            # Extract the red, green, and blue band values from the masked image

            vi = RGB2Dataset.calculate_vi(red, green, blue, selected_vi)
            # Calculate the selected VI values using the calculate_vi() method

            vi_values = vi.flatten()  # Flatten the image array to a 1D array
            vi_values = vi_values[~np.isnan(vi_values)]  # Keep only the non-NaN VI values
            threshold = otsu(vi_values, False)  # Calculate the Otsu threshold for the VI values

            # Create a list containing the plot index, plot ID, Otsu threshold, minimum VI value, and maximum VI value
            list_data = [polygon_idx, plot_name, threshold, min(vi_values), max(vi_values)]
            data_for_window.append(list_data)
            # Append the list to the data_for_window list

        return data_for_window

    def dataset_extraction_manu_plot(self, selected_vi, threshold, plot_index, hist_or_cc):
        """
        This method visualizes either the histogram of vegetation index (VI) values or the vegetation pixels for a selected
        plot, VI, and threshold.

        Args:
            self: The instance of the RGB2Dataset class.
            selected_vi (str): The name of the selected vegetation index.
            threshold (float): The threshold value for the selected VI.
            plot_index (int): The index of the plot to visualize.
            hist_or_cc (bool): If True, visualize the histogram of VI values; if False, visualize the vegetation pixels.
        """
        img = io.BytesIO()
        # Extract the geometry of the selected polygon
        polygon_geometry = self.gdf.iloc[plot_index]['geometry']
        plot_name = self.gdf.iloc[plot_index]['Plot_ID']
        # Get the geometry and plot ID of the selected plot from the shapefile

        # Clip the image to the selected polygon's geometry and specify the output CRS
        clipped_image, clipped_transform = mask(self.src, [polygon_geometry], crop=True)
        # Clip the input image to the selected plot's geometry using rasterio.mask.mask()

        # Mask black background
        mask_black = (clipped_image <= 0)
        masked_img = np.where(mask_black, np.nan, clipped_image)
        # Set pixel values <= 0 (black background) to NaN

        red = masked_img[0, :, :]
        green = masked_img[1, :, :]
        blue = masked_img[2, :, :]
        # Extract the red, green, and blue band values from the masked image

        # Calculate the VI values
        vi = RGB2Dataset.calculate_vi(red, green, blue, selected_vi)
        # Calculate the selected VI values using the calculate_vi() method

        title = selected_vi
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
        # Create a new figure and axis with a size of 5x5 inches

        if hist_or_cc:
            # If hist_or_cc is True, visualize the histogram of VI values
            vi_values = vi.flatten()  # Flatten the image array to a 1D array
            vi_values = vi_values[~np.isnan(vi_values)]  # Keep only the non-NaN VI values
            plot_VI_hist(ax, title, vi_values, plot_name)  # Plot the histogram of VI values

            plt.suptitle(f'{plot_name}:vegetation index hist')
            plt.tight_layout()
            img.seek(0)  # Reset the BytesIO object's position to the beginning
            # Save the plot to the BytesIO object instead of showing it
            plt.savefig(img, format='png')

        else:
            # If hist_or_cc is False, visualize the vegetation pixels
            VI_selected = vi
            threshold_selected = threshold

            # Apply thresholds to create a mask
            mask_selected = VI_selected < threshold_selected
            if title == 'ExR' or title == "CIVE":
                mask_selected = VI_selected >= threshold_selected
            # For ExR and CIVE, the mask condition is reversed

            VI_masked = np.where(mask_selected, np.nan, VI_selected)  # Mask the non-vegetation pixels
            vegetation_pixels = np.sum(~np.isnan(VI_masked))  # Count the number of vegetation pixels
            total_pixels = np.prod(VI_masked.shape) - np.sum(np.isnan(VI_selected))  # Count the total number of pixels
            vegetation_cover_ratio = vegetation_pixels / total_pixels  # Calculate the vegetation cover ratio

            # Plot vegetation pixels on the original image
            pure_soil_pixel_coords = [(col, row) for row, col in zip(*np.where(~np.isnan(VI_masked)))]
            crs_coords = [(col, row) for col, row in pure_soil_pixel_coords]  # Get the CRS coordinates of the vegetation pixels

            # Plot the scatter plot for the vegetation pixel coordinates in CRS coordinates
            crs_x_coords, crs_y_coords = zip(*crs_coords)
            ax.scatter(crs_x_coords, crs_y_coords, c='red', marker='x', s=1, label='Pure veg Pixels')

            # Display the clipped image and set the title with the vegetation cover ratio
            ax.imshow(clipped_image.transpose(1, 2, 0), cmap='RdYlGn', aspect=1)
            ax.axis('off')
            ax.set_title(f'{title} : CC={vegetation_cover_ratio}')

            plt.suptitle(f'{plot_name}:vegetation pixels with threshold {threshold}')
            plt.tight_layout()
            img.seek(0)  # Reset the BytesIO object's position to the beginning
            # Save the plot to the BytesIO object instead of showing it
            plt.savefig(img, format='png')

        # Return the BytesIO object
        return img

    def dataset_extraction_manu_ok(self, selected_vi, selected_threshold):
        """
        This method calculates the canopy cover (CC), mean values of red, green, and blue bands, and vegetation indices (VIs)
        for each plot in the shapefile based on the selected VI and thresholds. The results are saved in an Excel file.

        Args:
            self: The instance of the RGB2Dataset class.
            selected_vi (str): The name of the selected vegetation index.
            selected_threshold (list): A list of threshold values for the selected VI, one for each plot.
        """
        # Need to calculate canopy cover (cc), red_mean, green_mean, blue_mean, and vegetation indices (vi)

        # Create output directory using proper path joining
        manu_extract_dir = os.path.join(self.output_dir, 'Excel_' + self.filename)
        if not os.path.exists(manu_extract_dir):
            os.makedirs(manu_extract_dir)

        cc = []
        red_mean = []
        green_mean = []
        blue_mean = []
        # Initialize empty lists to store the canopy cover, mean values of red, green, and blue bands for each plot

        # Loop through the selected polygons and save each clipped image separately
        for polygon_idx, polygon in enumerate(self.gdf.geometry):
            # Extract the geometry of the selected polygon
            polygon_geometry = self.gdf.iloc[polygon_idx]['geometry']

            # Clip the image to the selected polygon's geometry and specify the output CRS
            clipped_image, clipped_transform = mask(self.src, [polygon_geometry], crop=True)
            # Clip the input image to the current plot's geometry using rasterio.mask.mask()

            # Clear out the background fill
            mask_black = (clipped_image <= 0)
            masked_img = np.where(mask_black, np.nan, clipped_image)
            # Set pixel values <= 0 (black background) to NaN

            red = masked_img[0, :, :]
            green = masked_img[1, :, :]
            blue = masked_img[2, :, :]
            # Extract the red, green, and blue band values from the masked image

            title = selected_vi
            # VI (vegetation index)
            vi = RGB2Dataset.calculate_vi(red, green, blue, selected_vi)
            # Calculate the selected VI values using the calculate_vi() method

            threshold = selected_threshold[polygon_idx]
            # Get the threshold value for the current plot from the selected_threshold list

            var_tuple = {'title': title, 'vi': vi, 'threshold': threshold, 'red': red, 'green': green, 'blue': blue}
            output = RGB2Dataset.calculate_canpoy_cover(var_tuple)  # Calculate the vegetation ratio
            # Calculate the canopy cover and other statistics using the calculate_canpoy_cover() method

            cc.append(output['cc'])
            red_mean.append(output['red'])
            green_mean.append(output['green'])
            blue_mean.append(output['blue'])
            # Append the canopy cover, mean values of red, green, and blue bands to the respective lists

        VIs = RGB2Dataset.calculate_vi(np.array(red_mean).astype(float), np.array(green_mean).astype(float),
                                        np.array(blue_mean).astype(float), None)
        # Calculate the vegetation indices using the mean values of red, green, and blue bands

        data_list = [('Plot_ID', self.gdf['Plot_ID']), (f'CC_{selected_vi}', cc), ('red', red_mean), ('green', green_mean),
                    ('blue', blue_mean)] + VIs
        # Create a list of tuples containing the column names and data for the DataFrame

        data_dict = {col_name: data for col_name, data in data_list}
        # Convert the list of tuples to a dictionary

        df = pd.DataFrame(data_dict)
        # Create a DataFrame from the dictionary

        # Write to the Excel file
        excel_filepath = os.path.join(manu_extract_dir, f'dataset_{selected_vi}.xlsx')
        df.to_excel(excel_filepath, index=False)
        # Save the DataFrame to an Excel file with the specified filename

        print('Dataset saved!!')

############################ MS ######################################
# Mutispectral image class:
class MS2Dataset():
    def __init__(self,src_r,src_g,src_b,src_re,src_nir,gdf,output_dir,clip_dir,is_stacked):
        self.src_r = src_r
        self.src_g = src_g
        self.src_b = src_b
        self.src_re = src_re
        self.src_nir = src_nir
        self.gdf = gdf
        self.is_stacked = is_stacked
        self.output_dir = current_app.config['GI_DIRS']['MS_OUTPUT_DIR']
        self.clip_dir = os.path.join(output_dir, 'shape_clip')

    def mask_black(self, polygon_geometry):
        """
        This method clips the input multispectral bands (red, green, blue, rededge, and NIR) to the specified polygon geometry,
        and masks out the black background pixels (pixel values <= 0) by setting them to NaN.

        Args:
            self: The instance of the MS2Dataset class.
            polygon_geometry: The geometry of the polygon to clip the bands to.

        Returns:
            A tuple containing the masked red, green, blue, rededge, and NIR band arrays.
        """
        # Clip the image to the selected polygon's geometry and specify the output CRS
        clipped_image_r, clipped_transform_r = mask(self.src_r, [polygon_geometry], crop=True)
        # Clip the red band (self.src_r) to the specified polygon geometry using rasterio.mask.mask()
        # The 'crop=True' parameter crops the output raster to the extent of the input geometry
        clipped_image_g, clipped_transform_g = mask(self.src_g, [polygon_geometry], crop=True)
        # Clip the green band (self.src_g) to the specified polygon geometry
        clipped_image_b, clipped_transform_b = mask(self.src_b, [polygon_geometry], crop=True)
        # Clip the blue band (self.src_b) to the specified polygon geometry
        clipped_image_re, clipped_transform_re = mask(self.src_re, [polygon_geometry], crop=True)
        # Clip the rededge band (self.src_re) to the specified polygon geometry
        clipped_image_nir, clipped_transform_nir = mask(self.src_nir, [polygon_geometry], crop=True)
        # Clip the NIR band (self.src_nir) to the specified polygon geometry
        clipped_images = [('red', clipped_image_r), ('green', clipped_image_g), ('blue', clipped_image_b),
                        ('rededge', clipped_image_re), ('nir', clipped_image_nir)]
        # Create a list of tuples containing the band names and their corresponding clipped band arrays

        # Clear out the background fill
        for name, clipped_image in clipped_images:
            mask_black = (clipped_image <= 0)
            # Create a boolean mask where True represents pixels with values <= 0 (black background)
            masked_img = np.where(mask_black, np.nan, clipped_image)
            # Set the black background pixels (values <= 0) to NaN using np.where()
            if name == 'red':
                red = masked_img[0, :, :]
                # Extract the masked red band array from the first channel of the masked image
            elif name == 'green':
                green = masked_img[0, :, :]
                # Extract the masked green band array from the first channel of the masked image
            elif name == 'blue':
                blue = masked_img[0, :, :]
                # Extract the masked blue band array from the first channel of the masked image
            elif name == 'rededge':
                rededge = masked_img[0, :, :]
                # Extract the masked rededge band array from the first channel of the masked image
            elif name == 'nir':
                nir = masked_img[0, :, :]
                # Extract the masked NIR band array from the first channel of the masked image

        return red, green, blue, rededge, nir
        # Return the masked red, green, blue, rededge, and NIR band arrays as a tuple

    @staticmethod
    def calculate_canpoy_cover(input_dir):
        
        VI_selected = input_dir['vi']
        selected_threshold = input_dir['threshold']
        # title = input_dir['title']
        red = input_dir['red']
        green = input_dir['green']
        blue = input_dir['blue']
        rededge = input_dir['re']
        nir = input_dir['nir']
        # Apply thresholds to create a mask
        mask =  VI_selected < selected_threshold

        #  if title == 'ExR' or title == "CIVE":
        #       mask =  VI_selected >= selected_threshold

        VI_masked = np.where(mask,np.nan,VI_selected) # mask the non veg pixels

        vegetation_pixels = np.sum(~np.isnan(VI_masked)) # extrat the veg pixels

        total_pixels = np.prod(VI_masked.shape)-np.sum(np.isnan(VI_selected)) # the total pixels

        vegetation_cover_ratio = vegetation_pixels / total_pixels  # canopy cover

        red_masked = np.where(mask,np.nan,red)
        green_masked = np.where(mask,np.nan,green)
        blue_masked = np.where(mask,np.nan,blue)
        rededge_masked = np.where(mask,np.nan,rededge)
        nir_masked = np.where(mask,np.nan,nir)

        red_masked = red_masked[~np.isnan(red_masked)]
        green_masked = green_masked[~np.isnan(green_masked)]
        blue_masked = blue_masked[~np.isnan(blue_masked)]
        rededge_masked = rededge_masked[~np.isnan(rededge_masked)]
        nir_masked = nir_masked[~np.isnan(nir_masked)]

        red_mean = np.mean(red_masked)
        green_mean = np.mean(green_masked)
        blue_mean =np.mean(blue_masked)
        rededge_mean =np.mean(rededge_masked)
        nir_mean =np.mean(nir_masked)

        return {'cc':vegetation_cover_ratio,'red':red_mean,'green':green_mean,'blue':blue_mean,'re':rededge_mean,'nir':nir_mean}

    @staticmethod
    def calculate_VI(red,green,blue,rededge,nir,selected_vi):
        # VI(vegetation index)
        # Constants for EVI calculation
        L = 0.16  # Canopy background adjustment; gsavi
        x=0.08 # atsavi
        a=1.43 # atsavi, savi2
        b=0.01 # atsavi, savi2

        # Calculate VIs
        if selected_vi != None:
            match selected_vi:
                case 'NDVI':
                    vi = (nir - red) / (nir + red)
                case 'GNDVI':
                    vi = (nir - green) / (nir + green)
                case 'DVI':
                    vi = (nir - red)
                case 'EVI':
                    vi = (2.5*(nir-red))/((nir+(6*red)-(7.5*blue))+1) #evi
                case 'EVI2':
                    vi = (2.5*(nir-red))/((nir+(2.4*red))+1)
                case 'ARI':
                    vi = (1/green)-(1/rededge)
                case 'MARI':
                    vi = (((1/green)-(1/rededge))*nir)
                case 'CHLRE': 
                    vi = (nir/rededge)-1
                case 'CHLGR':
                    vi = (nir/green)-1 
                case 'SAVI':
                    savi= (1.5*(nir - red))/(nir+red+0.5) # SAVI
                case 'SAVI2':
                    savi2= nir/(red+(b/a)) # SAVI2
                case 'MSAVI':
                    vi= ((2*nir)+1-np.sqrt(((2*nir)+1)**2-(8*(nir-red))))/2 # MSAVI
                case 'OSAVI':
                    vi=(1+L)*((nir-green)/(nir+green+L)) #OSAVI 
                case 'TSAVI':
                    vi = ((a*(nir-(a*red)-b))/((a*nir)+red-(a*b)+(x*(1+a**2)))) #or TSAVI
                case 'MSR':
                    vi = ((nir/red)-1)/np.sqrt(((nir/red)-1))
                case 'MCARI': 
                    vi = ((rededge - red) - 0.2 * (rededge - green)) * (rededge / red)
                case 'MACRIDIOSAVI':
                    osavi = (1+L)*((nir-green)/(nir+green+L))
                    mcari = ((rededge - red) - 0.2 * (rededge - green)) * (rededge / red)
                    vi = mcari/osavi
                case 'PVI':
                    vi = (nir-(a*red)-b)/(np.sqrt(1+a**2))
                case 'SR':
                    vi = nir/red
                case 'WDRI': 
                    vi = (0.3*(nir-red))/(0.3*(nir+red))   
            return vi                       

        else:
            ndvi = (nir - red) / (nir + red) # ndvi
            gndvi =(nir - green) / (nir + green)
            dvi= (nir - red)
            evi = (2.5*(nir-red))/((nir+(6*red)-(7.5*blue))+1) #evi
            evi2 = (2.5*(nir-red))/((nir+(2.4*red))+1)
            ari= (1/green)-(1/rededge)
            mari = (((1/green)-(1/rededge))*nir)
            chlre= (nir/rededge)-1
            chlgr= (nir/green)-1 
            savi= (1.5*(nir - red))/(nir+red+0.5) # SAVI
            savi2= nir/(red+(b/a)) # SAVI2
            msavi= ((2*nir)+1-np.sqrt(((2*nir)+1)**2-(8*(nir-red))))/2 # MSAVI
            osavi=(1+L)*((nir-green)/(nir+green+L)) #OSAVI 
            tsavi = ((a*(nir-(a*red)-b))/((a*nir)+red-(a*b)+(x*(1+a**2)))) #or TSAVI
            msr = ((nir/red)-1)/np.sqrt(((nir/red)-1))
            mcari = ((rededge - red) - 0.2 * (rededge - green)) * (rededge / red)
            mcaridiosavi = mcari/osavi
            pvi = (nir-(a*red)-b)/(np.sqrt(1+a**2))
            sr=nir/red
            wdri= (0.3*(nir-red))/(0.3*(nir+red))

            VIs = [('NDVI',ndvi), ('GNDVI',gndvi), ('DVI',dvi), ('EVI',evi), ('EVI2',evi2), ('ARI',ari),('MARI',mari), ('CHLRE',chlre), ('CHLGR',chlgr), 
                ('SAVI',savi), ('SAVI2',savi2), ('MSAVI',msavi), ('OSAVI',osavi), ('TSAVI',tsavi), ('MSR',msr), ('MCARI',mcari), 
                ('MCARIDIOSAVI',mcaridiosavi), ('PVI',pvi), ('SR',sr), ('WDRI',wdri)]
            
            return VIs

    # clip shape for each band
    def clip_rasterio_shape(self):
        """
        Input: 
            image_path: path to image
            shapefile_path: path to your shapefile
        """
        # Create output directory using proper path joining
        output_path_clip = os.path.join(self.output_dir, 'shape_clip')
        
        # Clean up existing directory if it exists
        if os.path.exists(output_path_clip):
            shutil.rmtree(output_path_clip)
            time.sleep(0.5)  # Give OS time to complete deletion
        
        # Create fresh directory
        os.makedirs(output_path_clip, exist_ok=True)

        all_src = [('red', self.src_r), ('green', self.src_g), ('blue', self.src_b), 
                ('re', self.src_re), ('nir', self.src_nir)]

        for band_name, band_src in all_src:    
            # Create band-specific output directory
            output_path = os.path.join(output_path_clip, band_name)
            os.makedirs(output_path, exist_ok=True)

            # Clip image to the shapefile geometry
            clipped_image, clipped_transform = mask(band_src, self.gdf.geometry, crop=True)

            # Specify output path for each individual band using the image name
            clipped_image_path = os.path.join(output_path, f'{band_name}-clipped.tif')

            # Create a new raster file with the same dimensions as the clipped band
            with rasterio.open(clipped_image_path, 'w', driver='GTiff', width=clipped_image.shape[2], height=clipped_image.shape[1], count=clipped_image.shape[0], dtype=clipped_image.dtype, crs=band_src.crs, transform=clipped_transform) as dst:
                dst.write(clipped_image)
            print(f'clipped image {band_name}-clipped.tif saved to {clipped_image_path}')
            
            # Print shape and CRS (sanity check)
            print(f'Shape: {clipped_image.shape}, CRS: {band_src.crs}')
            
            # loop through the selected polygons and save each clipped image separately
            for polygon_idx, polygon in enumerate(self.gdf.geometry):
                # Extract the geometry of the selected polygon
                polygon_geometry = self.gdf.iloc[polygon_idx]['geometry']
                plot_name = self.gdf.iloc[polygon_idx]['Plot_ID']   # here we cannot fix it, should be changable.

                # Clip the image to the selected polygon's geometry and specify the output CRS
                clipped_image, clipped_transform = mask(band_src, [polygon_geometry], crop=True)
                clipped_meta = band_src.meta.copy()
                clipped_meta['crs'] = {'init': f'epsg:{self.gdf.crs}'}

                # output file path for the clipped image using the band and location index
                output_image_path = os.path.join(output_path, f'{plot_name}.tif')

                # Create a new raster file with the same dimensions as the clipped image
                with rasterio.open(output_image_path, 'w', driver='GTiff', width=clipped_image.shape[2], height=clipped_image.shape[1], count=clipped_image.shape[0], dtype=clipped_image.dtype, crs=band_src.crs, transform=clipped_transform) as dst:
                    dst.write(clipped_image)

            print("Clipped images saved in the output directories for each plot.")

        return True

    def dataset_extraction_auto(self, target_df):
        """
        This function extracts datasets from multispectral images based on the provided target dataframe.
        It calculates various vegetation indices and canopy cover values for each plot in the input shapefile.
        The datasets with the highest correlation to the target variable are saved as Excel files.
        """
        # Check for common Plot_ID values
        if not bool(set(self.gdf['Plot_ID']).intersection(set(target_df['Plot_ID']))):
            print(f"You may need to name your Plot_ID column by this format:\n {self.gdf['Plot_ID']}")
            return

        # Create output directory using proper path joining
        auto_extract_dir = os.path.join(self.output_dir, 'Excel')
        if not os.path.exists(auto_extract_dir):
            os.makedirs(auto_extract_dir)

        plot_num = self.gdf.shape[0]
        col_name = ['NDVI', 'GNDVI', 'DVI', 'EVI', 'EVI2', 'ARI', 'MARI', 'CHLRE', 'CHLGR', 
                    'SAVI', 'SAVI2', 'MSAVI', 'OSAVI', 'TSAVI', 'MSR', 'MCARI', 'MCARIDIOSAVI', 
                    'PVI', 'SR', 'WDRI']
        column_num = len(col_name)

        cc_all = [[None] * column_num for _ in range(plot_num)]
        red_all = [[None] * column_num for _ in range(plot_num)]
        green_all = [[None] * column_num for _ in range(plot_num)]
        blue_all = [[None] * column_num for _ in range(plot_num)]
        nir_all = [[None] * column_num for _ in range(plot_num)]
        rededge_all = [[None] * column_num for _ in range(plot_num)]

        # Use pre-clipped images directory
        clip_dir = os.path.join(self.output_dir, 'shape_clip')

        # Process each plot using pre-clipped band images
        for polygon_idx, (plot_id, geometry) in enumerate(zip(self.gdf['Plot_ID'], self.gdf.geometry)):
            # Load each pre-clipped band
            band_sources = {
                'r_band': rasterio.open(os.path.join(clip_dir, 'r_band', f'{plot_id}.tif')),
                'g_band': rasterio.open(os.path.join(clip_dir, 'g_band', f'{plot_id}.tif')),
                'b_band': rasterio.open(os.path.join(clip_dir, 'b_band', f'{plot_id}.tif')),
                're_band': rasterio.open(os.path.join(clip_dir, 're_band', f'{plot_id}.tif')),
                'nir_band': rasterio.open(os.path.join(clip_dir, 'nir_band', f'{plot_id}.tif'))
            }

            # Clip each band using the polygon geometry
            clipped_bands = {}
            for band_name, src in band_sources.items():
                clipped_image, clipped_transform = mask(src, [geometry], crop=True)
                mask_black = (clipped_image <= 0)
                clipped_bands[band_name] = np.where(mask_black, np.nan, clipped_image)[0,:,:]

            # Calculate vegetation indices
            VIs = MS2Dataset.calculate_VI(clipped_bands['r_band'], clipped_bands['g_band'], 
                                        clipped_bands['b_band'], clipped_bands['re_band'], 
                                        clipped_bands['nir_band'], None)
            cc = []
            red_mean = []
            green_mean = []
            blue_mean = []
            rededge_mean = []
            nir_mean = []

            for (title, vi) in VIs:
                vi_values = vi.flatten()
                vi_values = vi_values[~np.isnan(vi_values)]
                threshold = otsu(vi_values, False)
                var_tuple = {'vi': vi, 'threshold': threshold, 
                            'red': clipped_bands['r_band'],
                            'green': clipped_bands['g_band'], 
                            'blue': clipped_bands['b_band'],
                            're': clipped_bands['re_band'], 
                            'nir': clipped_bands['nir_band']}
                output = MS2Dataset.calculate_canpoy_cover(var_tuple)
                cc.append(output['cc'])
                red_mean.append(output['red'])
                green_mean.append(output['green'])
                blue_mean.append(output['blue'])
                rededge_mean.append(output['re'])
                nir_mean.append(output['nir'])

            cc_all[polygon_idx] = cc
            red_all[polygon_idx] = red_mean
            green_all[polygon_idx] = green_mean
            blue_all[polygon_idx] = blue_mean
            rededge_all[polygon_idx] = rededge_mean
            nir_all[polygon_idx] = nir_mean

            # Close all band sources
            for src in band_sources.values():
                src.close()

        # Create separate dataframes for each band
        cc_df = pd.DataFrame(cc_all, columns=col_name)
        cc_df.insert(0, column='Plot_ID', value=self.gdf['Plot_ID'])

        red_df = pd.DataFrame(red_all, columns=col_name)
        red_df.insert(0, column='Plot_ID', value=self.gdf['Plot_ID'])

        green_df = pd.DataFrame(green_all, columns=col_name)
        green_df.insert(0, column='Plot_ID', value=self.gdf['Plot_ID'])

        blue_df = pd.DataFrame(blue_all, columns=col_name)
        blue_df.insert(0, column='Plot_ID', value=self.gdf['Plot_ID'])

        re_df = pd.DataFrame(rededge_all, columns=col_name)
        re_df.insert(0, column='Plot_ID', value=self.gdf['Plot_ID'])

        nir_df = pd.DataFrame(nir_all, columns=col_name)
        nir_df.insert(0, column='Plot_ID', value=self.gdf['Plot_ID'])

        # Merge the canopy cover, red, green, blue, NIR, and red-edge dataframes with the target dataframe
        merged_df_cc = pd.merge(cc_df, target_df, how='inner', left_on='Plot_ID', right_on='Plot_ID')
        merged_df_red = pd.merge(red_df, target_df, how='inner', left_on='Plot_ID', right_on='Plot_ID')
        merged_df_green = pd.merge(green_df, target_df, how='inner', left_on='Plot_ID', right_on='Plot_ID')
        merged_df_blue = pd.merge(blue_df, target_df, how='inner', left_on='Plot_ID', right_on='Plot_ID')
        merged_df_re = pd.merge(re_df, target_df, how='inner', left_on='Plot_ID', right_on='Plot_ID')
        merged_df_nir = pd.merge(nir_df, target_df, how='inner', left_on='Plot_ID', right_on='Plot_ID')

        # Calculate the correlation between each vegetation index and the target variable
        target_col_name = merged_df_cc.columns[-1]
        correlations = {}
        for col in merged_df_cc.columns[1:-1]:
            correlation = merged_df_cc[col].corr(merged_df_cc[target_col_name])  # Calculate the correlation with the target variable
            correlations[col] = correlation
        print(correlations)

        # Find the vegetation index(es) with the highest correlation to the target variable
        max_corr_value = max(correlations.values())  # Get the maximum correlation value
        print(f" The max value is: {max_corr_value}")
        max_corr_columns = [col for col, correlation in correlations.items() if correlation == max_corr_value]  # Get the column names with the maximum correlation
        round_max_corr = r'{:.2f}'.format(float(max_corr_value))
        print(max_corr_columns, max_corr_value)

        # Create and save Excel files for the vegetation index(es) with the highest correlation
        for col in max_corr_columns:
            df_new = merged_df_cc.iloc[:, :1]  # Create a new dataframe with the 'Plot_ID' column
            df_new[target_col_name] = merged_df_cc[target_col_name]  # Add the target variable column
            df_new['CC'] = merged_df_cc[f'{col}']  # Add the canopy cover column for the current vegetation index

            # Get the red, green, blue, NIR, and red-edge values for the current vegetation index
            red = merged_df_red[f'{col}']
            green = merged_df_green[f'{col}']
            blue = merged_df_blue[f'{col}']
            rededge = merged_df_re[f'{col}']
            nir = merged_df_nir[f'{col}']

            # Calculate vegetation indices using the red, green, blue, NIR, and red-edge values
            VIs = MS2Dataset.calculate_VI(red, green, blue, rededge, nir, None)

            # Create a list of dataframes for red, green, blue, NIR, red-edge, and vegetation indices
            data_df = [('red', red), ('green', green), ('blue', blue), ('rededge', rededge), ('nir', nir)] + VIs

            # Add the red, green, blue, NIR, red-edge, and vegetation index columns to the new dataframe
            for colname, coldata in data_df:
                df_new[colname] = coldata

            # Save the new dataframe as an Excel file
            excel_filepath = os.path.join(auto_extract_dir, f'dataset_{col}_{round_max_corr}.xlsx')
            df_new.to_excel(excel_filepath, index=False)

    def visualization_plot(self, plot_index, is_hist, selected_band_show):
        """
        This function visualizes the selected plot from the input shapefile.
        It can either display histograms of vegetation indices or show the original image with vegetation pixels highlighted.
        """
        img = io.BytesIO
        
        # Extract the geometry and plot name for the selected polygon
        polygon_geometry = self.gdf.iloc[plot_index]['geometry']
        plot_name = self.gdf.iloc[plot_index]['Plot_ID']

        # Mask out the background (non-vegetation pixels) and get the red, green, blue, red-edge, and NIR bands
        red, green, blue, rededge, nir = self.mask_black(polygon_geometry)

        # Calculate vegetation indices for the selected plot
        VIs = MS2Dataset.calculate_VI(red, green, blue, rededge, nir, None)

        if is_hist:
            # If is_hist is True, plot histograms of vegetation indices
            fig, axs = plt.subplots(nrows=5, ncols=4, figsize=(12, 8))
            axs = np.ravel(axs)

            for ax, (title, vi) in zip(axs, VIs):
                vi_values = vi.flatten()  # Flatten the image array to a 1D array
                vi_values = vi_values[~np.isnan(vi_values)]  # Keep only the non-NaN vegetation index values
                plot_VI_hist(ax, title, vi_values, plot_name)  # Plot histogram of pixel values for the current vegetation index
            plt.suptitle('vegetation index hist')
        else:
            # If is_hist is False, show the original image with vegetation pixels highlighted
            match selected_band_show:
                # Select the band to display based on the user's choice
                case 'red':
                    selected_band = red
                case 'green':
                    selected_band = green
                case 'blue':
                    selected_band = blue
                case 'rededge':
                    selected_band = rededge
                case 'nir':
                    selected_band = nir
                case _:
                    selected_band = nir

            threshold_all = []

            # Loop through each vegetation index
            for (title, vi) in VIs:
                vi_values = vi.flatten()  # Flatten the image array to a 1D array
                vi_values = vi_values[~np.isnan(vi_values)]  # Keep only the non-NaN vegetation index values
                threshold = otsu(vi_values, False)  # Calculate the Otsu threshold for the current vegetation index
                threshold_all.append(threshold)

            # Create a list of tuples containing the vegetation index, image, and threshold
            VIs_th = [(v[0], v[1], t) for v, t in zip(VIs, threshold_all)]

            fig, axs = plt.subplots(nrows=5, ncols=4, figsize=(12, 8))
            axs = np.ravel(axs)

            for ax, (title, vi, threshold) in zip(axs, VIs_th):
                print(vi, threshold)
                VI_selected = vi
                threshold_selected = threshold

                # Apply the threshold to create a mask for vegetation pixels
                mask_selected = VI_selected < threshold_selected
                VI_masked = np.where(mask_selected, np.nan, VI_selected)  # Mask the non-vegetation pixels

                # Calculate the vegetation cover ratio
                vegetation_pixels = np.sum(~np.isnan(VI_masked))  # Count the number of vegetation pixels
                total_pixels = np.prod(VI_masked.shape) - np.sum(np.isnan(VI_selected))  # Count the total number of pixels
                vegetation_cover_ratio = vegetation_pixels / total_pixels  # Calculate the vegetation cover ratio

                # Get the coordinates of the vegetation pixels
                pure_soil_pixel_coords = [(col, row) for row, col in zip(*np.where(~np.isnan(VI_masked)))]
                crs_coords = [(col, row) for col, row in pure_soil_pixel_coords]  # Get the CRS coordinates of the vegetation pixels

                # Plot the vegetation pixels on the original image
                crs_x_coords, crs_y_coords = zip(*crs_coords)
                ax.scatter(crs_x_coords, crs_y_coords, c='red', marker='x', s=1, label='Pure veg Pixels')

                # Display the original image and set the labels and colorbar
                ax.imshow(selected_band, cmap='RdYlGn', aspect=1)
                ax.axis('off')
                ax.set_title(f'{title} : CC={vegetation_cover_ratio}')

            plt.suptitle(f'{plot_name}:vegetation pixels')

        plt.tight_layout()
        img.seek(0)  # Reset the BytesIO object's position to the beginning
        # Save the plot to the BytesIO object instead of showing it
        plt.savefig(img, format='png')

        return img

    def dataset_extraction_manu_threshold(self, selected_vi):
        """
        This function calculates the Otsu threshold for a selected vegetation index (VI) for each plot in the input shapefile.
        It returns a list of data containing the polygon index, plot name, threshold, minimum VI value, and maximum VI value.
        """
        data_for_window = []

        # Loop through each polygon in the input shapefile
        for polygon_idx, polygon in enumerate(self.gdf.geometry):
            # Extract the geometry and plot name for the current polygon
            polygon_geometry = self.gdf.iloc[polygon_idx]['geometry']
            plot_name = self.gdf.iloc[polygon_idx]['Plot_ID']

            # Mask out the background (non-vegetation pixels) and get the red, green, blue, red-edge, and NIR bands
            red, green, blue, rededge, nir = self.mask_black(polygon_geometry)

            # Calculate the selected vegetation index for the current polygon
            vi = MS2Dataset.calculate_VI(red, green, blue, rededge, nir, selected_vi)

            # Flatten the vegetation index image array to a 1D array
            vi_values = vi.flatten()

            # Keep only the non-NaN vegetation index values
            vi_values = vi_values[~np.isnan(vi_values)]

            # Calculate the Otsu threshold for the vegetation index values
            threshold = otsu(vi_values, False)

            # Create a list containing the polygon index, plot name, threshold, minimum VI value, and maximum VI value
            list_data = [polygon_idx, plot_name, threshold, min(vi_values), max(vi_values)]

            # Append the list to the data_for_window list
            data_for_window.append(list_data)

        return data_for_window

    def dataset_extraction_manu_plot(self, selected_vi, threshold, plot_index, hist_or_cc, selected_band_show):
        """
        This function visualizes the selected vegetation index (VI) for a specific plot in the input shapefile.
        It can either display a histogram of the VI values or show the original image with vegetation pixels highlighted.
        """
        img = io.BytesIO()

        # Extract the geometry and plot name for the selected polygon
        polygon_geometry = self.gdf.iloc[plot_index]['geometry']
        plot_name = self.gdf.iloc[plot_index]['Plot_ID']

        # Mask out the background (non-vegetation pixels) and get the red, green, blue, red-edge, and NIR bands
        red, green, blue, rededge, nir = self.mask_black(polygon_geometry)

        # Calculate the selected vegetation index for the selected plot
        vi = MS2Dataset.calculate_VI(red, green, blue, rededge, nir, selected_vi)

        title = selected_vi
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))

        if hist_or_cc:
            # If hist_or_cc is True, plot a histogram of the VI values
            vi_values = vi.flatten()  # Flatten the image array to a 1D array
            vi_values = vi_values[~np.isnan(vi_values)]  # Keep only the non-NaN VI values
            plot_VI_hist(ax, title, vi_values, plot_name)  # Plot histogram of pixel values
            plt.suptitle(f'{plot_name}:vegetation index hist')
            plt.tight_layout()
            img.seek(0)  # Reset the BytesIO object's position to the beginning
            # Save the plot to the BytesIO object instead of showing it
            plt.savefig(img, format='png')
        else:
            # If hist_or_cc is False, show the original image with vegetation pixels highlighted
            VI_selected = vi
            threshold_selected = threshold

            # Select the band to display based on the user's choice
            match selected_band_show:
                case 'red':
                    selected_band = red
                case 'green':
                    selected_band = green
                case 'blue':
                    selected_band = blue
                case 'rededge':
                    selected_band = rededge
                case 'nir':
                    selected_band = nir
                case _:
                    selected_band = nir

            # Apply the threshold to create a mask for vegetation pixels
            mask_selected = VI_selected < threshold_selected
            VI_masked = np.where(mask_selected, np.nan, VI_selected)  # Mask the non-vegetation pixels

            # Calculate the vegetation cover ratio
            vegetation_pixels = np.sum(~np.isnan(VI_masked))  # Count the number of vegetation pixels
            total_pixels = np.prod(VI_masked.shape) - np.sum(np.isnan(VI_selected))  # Count the total number of pixels
            vegetation_cover_ratio = vegetation_pixels / total_pixels  # Calculate the vegetation cover ratio

            # Get the coordinates of the vegetation pixels
            pure_soil_pixel_coords = [(col, row) for row, col in zip(*np.where(~np.isnan(VI_masked)))]
            crs_coords = [(col, row) for col, row in pure_soil_pixel_coords]  # Get the CRS coordinates of the vegetation pixels

            # Plot the vegetation pixels on the original image
            crs_x_coords, crs_y_coords = zip(*crs_coords)
            ax.scatter(crs_x_coords, crs_y_coords, c='red', marker='x', s=1, label='Pure veg Pixels')

            # Display the original image and set the labels and colorbar
            ax.imshow(selected_band, cmap='RdYlGn', aspect=1)
            ax.axis('off')
            ax.set_title(f'{title} : CC={vegetation_cover_ratio}')

            plt.suptitle(f'{plot_name}:vegetation pixels with threshold {threshold}')
            plt.tight_layout()
            img.seek(0)  # Reset the BytesIO object's position to the beginning
            # Save the plot to the BytesIO object instead of showing it
            plt.savefig(img, format='png')
        
        return img

    def dataset_extraction_manu_ok(self, selected_vi, selected_threshold):
        """
        This function calculates canopy cover (CC), mean red, green, blue, red-edge, and NIR values, and vegetation indices (VIs)
        for the selected vegetation index and thresholds. It saves the resulting dataset as an Excel file.
        """
        # need to calculate cc, red_mean, green_mean, blue_mean, vi

        # Create output directory using proper path joining
        manu_extract_dir = os.path.join(self.output_dir, 'Excel')
        if not os.path.exists(manu_extract_dir):
            os.makedirs(manu_extract_dir)


        # Initialize lists to store canopy cover, red, green, blue, NIR, and red-edge values
        cc = []
        red_mean = []
        green_mean = []
        blue_mean = []
        rededge_mean = []
        nir_mean = []

        # Loop through each polygon in the input shapefile
        for polygon_idx, polygon in enumerate(self.gdf.geometry):
            # Extract the geometry of the selected polygon
            polygon_geometry = self.gdf.iloc[polygon_idx]['geometry']

            # Mask out the background (non-vegetation pixels) and get the red, green, blue, red-edge, and NIR bands
            red, green, blue, rededge, nir = self.mask_black(polygon_geometry)

            # Calculate the selected vegetation index for the current polygon
            vi = MS2Dataset.calculate_VI(red, green, blue, rededge, nir, selected_vi)
            threshold = selected_threshold[polygon_idx]  # Get the threshold for the current polygon

            # Create a dictionary with the vegetation index, threshold, and band values
            var_tuple = {'vi': vi, 'threshold': threshold, 'red': red, 'green': green, 'blue': blue, 're': rededge, 'nir': nir}
            output = MS2Dataset.calculate_canpoy_cover(var_tuple)  # Calculate the canopy cover and mean band values

            # Append the canopy cover and mean band values to the respective lists
            cc.append(output['cc'])
            red_mean.append(output['red'])
            green_mean.append(output['green'])
            blue_mean.append(output['blue'])
            rededge_mean.append(output['re'])
            nir_mean.append(output['nir'])

        # Calculate vegetation indices using the mean red, green, blue, NIR, and red-edge values
        VIs = MS2Dataset.calculate_VI(np.array(red_mean).astype(float), np.array(green_mean).astype(float),
                                    np.array(blue_mean).astype(float), np.array(rededge_mean).astype(float),
                                    np.array(nir_mean).astype(float), None)

        # Create a list of tuples containing column names and data
        data_list = [('Plot_ID', self.gdf['Plot_ID']), (f'CC_{selected_vi}', cc), ('red', red_mean), ('green', green_mean),
                    ('blue', blue_mean), ('rededge', rededge_mean), ('nir', nir_mean)] + VIs

        # Convert the list of tuples to a dictionary
        data_dict = {col_name: data for col_name, data in data_list}

        # Create a pandas DataFrame from the dictionary
        df = pd.DataFrame(data_dict)

        # Write the DataFrame to an Excel file
        excel_filepath = os.path.join(manu_extract_dir, f'dataset_{selected_vi}.xlsx')
        df.to_excel(excel_filepath, index=False)

        print('Dataset saved!!')

    def gdf_visualization(self):
        """
        This function visualizes the input shapefile (GeoDataFrame) by plotting the geometries and annotating each polygon with its corresponding Plot_ID.
        """
        # Plot the geometries from the GeoDataFrame with a figure size of 12x8 inches
        ax = self.gdf.geometry.plot(figsize=(12, 8))

        # Annotate each polygon with its Plot_ID
        self.gdf.apply(lambda x: ax.annotate(text=x['Plot_ID'], xy=x.geometry.centroid.coords[0], ha='center'), axis=1)

    def check_clipped(self, selected_band):
        """
        This function visualizes the clipped multispectral image for the selected band.
        It displays the clipped image with the shapefile boundaries and Plot_ID annotations.
        """
        img = io.BytesIO()
        
        # Select the appropriate rasterio source based on the selected band
        match selected_band:
            case 'red':
                src = self.src_r
            case 'green':
                src = self.src_g
            case 'blue':
                src = self.src_b
            case 'rededge':
                src = self.src_re
            case 'nir':
                src = self.src_nir
            case _:
                src = self.src_nir

        # Clip the selected band image to the shapefile geometries
        clipped_image, clipped_transform = mask(src, self.gdf.geometry, crop=True)

        # Replace negative values with NaN (to handle background fill)
        mask_black = clipped_image < 0
        image = np.where(mask_black, np.nan, clipped_image)

        # Create a GeoDataFrame with the clipped geometries
        gdf_clipped = gpd.GeoDataFrame({'geometry': [g for g in self.gdf.geometry]}, crs=self.gdf.crs)

        # Plot the clipped image with the CRS coordinates and shapefile boundaries
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # Display the clipped image
        im = rasterio.plot.show(image, transform=clipped_transform, ax=ax, cmap='viridis')

        # Plot the shapefile boundaries on top of the image
        gdf_clipped.boundary.plot(ax=ax, color='red')

        # Annotate each polygon with its Plot_ID
        self.gdf.apply(lambda x: ax.annotate(text=x['Plot_ID'], xy=x.geometry.centroid.coords[0], ha='center', color='red'), axis=1)

        # Set axis labels
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        plt.tight_layout()
        img.seek(0)  # Reset the BytesIO object's position to the beginning
        # Save the plot to the BytesIO object instead of showing it
        plt.savefig(img, format='png')

        # Return the BytesIO object
        return img

