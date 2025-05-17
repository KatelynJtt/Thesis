document.addEventListener('DOMContentLoaded', function() {
    
// UPLOAD IMAGES ---------------------------------------------------------------------------------------
    // Handle image file uploads
    const rgbImageInput = document.getElementById('rgbImageInput');
    const rgbImageUploadBtn = document.getElementById('rgbImageUploadBtn');

    if (rgbImageInput && rgbImageUploadBtn) {
        rgbImageUploadBtn.addEventListener('click', handleRGBImageUpload);
    } else {
        console.error('Unable to find the required elements in the DOM.');
    }

    // Variable to store the uploaded RGB image file
    let uploadedRGBImageFile;

    // Function to handle RGB image upload
    function handleRGBImageUpload() {
        const rgbImageInput = document.getElementById('rgbImageInput');
        const file = rgbImageInput.files[0];

        if (file) {
            uploadedRGBImageFile = file; // Store the file object

            sendFileToServer(file, '/geoimaging/upload_rgb_image')
                .then(data => {
                    const statusElement = document.getElementById('rgbImageUploadStatus');
                    if (data.success) {
                        statusElement.innerHTML = '&#10004;'; // Checkmark
                    } else {
                        statusElement.innerHTML = 'Error: ' + data.error;
                    }
                })
                .catch(error => {
                    const statusElement = document.getElementById('rgbImageUploadStatus');
                    statusElement.innerHTML = 'Error: ' + error;
                });
        } else {
            console.error('No RGB image file provided.');
            const statusElement = document.getElementById('rgbImageUploadStatus');
            statusElement.innerHTML = 'Error: No file provided';
        }
    }

    // Variables to store the uploaded file objects for each band
    let uploadedRBandFile;
    let uploadedGBandFile;
    let uploadedBBandFile;
    let uploadedREBandFile;
    let uploadedNIRBandFile;

    // Event listeners for band upload buttons
    const rBandUploadBtn = document.getElementById('rBandUploadBtn');
    rBandUploadBtn.addEventListener('click', () => handleBandUpload('r_band'));

    const gBandUploadBtn = document.getElementById('gBandUploadBtn');
    gBandUploadBtn.addEventListener('click', () => handleBandUpload('g_band'));

    const bBandUploadBtn = document.getElementById('bBandUploadBtn');
    bBandUploadBtn.addEventListener('click', () => handleBandUpload('b_band'));

    const reBandUploadBtn = document.getElementById('reBandUploadBtn');
    reBandUploadBtn.addEventListener('click', () => handleBandUpload('re_band'));

    const nirBandUploadBtn = document.getElementById('nirBandUploadBtn');
    nirBandUploadBtn.addEventListener('click', () => handleBandUpload('nir_band'));

    // Function to handle band upload
    function handleBandUpload(bandType) {
        const bandInput = document.getElementById(bandType);
        const file = bandInput.files[0];

        if (file) {
            // Store the file object based on the band type
            switch (bandType) {
                case 'r_band':
                    uploadedRBandFile = file;
                    break;
                case 'g_band':
                    uploadedGBandFile = file;
                    break;
                case 'b_band':
                    uploadedBBandFile = file;
                    break;
                case 're_band':
                    uploadedREBandFile = file;
                    break;
                case 'nir_band':
                    uploadedNIRBandFile = file;
                    break;
            }

            sendFileToServer(file, `/geoimaging/upload_ms_band?band_type=${bandType}`)
                .then(data => {
                    const statusElement = document.getElementById(`${bandType}UploadStatus`);
                    if (data.success) {
                        statusElement.innerHTML = '&#10004;'; // Checkmark
                    } else {
                        statusElement.innerHTML = 'Error: ' + data.error;
                    }
                })
                .catch(error => {
                    const statusElement = document.getElementById(`${bandType}UploadStatus`);
                    statusElement.innerHTML = 'Error: ' + error;
                });
        } else {
            console.error(`No ${bandType} file provided.`);
            const statusElement = document.getElementById(`${bandType}UploadStatus`);
            statusElement.innerHTML = 'Error: No file provided';
        }
    }

// UPLOAD SHAPEFILES ------------------------------------------------------------------------------------------------
    // Handle shapefile uploads
    const rgbShapefileUploadBtn = document.getElementById('rgbShapefileUploadBtn');
    rgbShapefileUploadBtn.addEventListener('click', () => {
        const rgbShapefileInput = document.getElementById('rgbShapefile');
        const files = rgbShapefileInput.files;
        if (files.length > 0) {
            uploadShapefile(files, 'rgb');
        } else {
            console.error('No RGB shapefile files provided.');
        }
    });

    // Event listener for MS shapefile upload
    const msShapefileUploadBtn = document.getElementById('msShapefileUploadBtn');
    msShapefileUploadBtn.addEventListener('click', () => {
        const msShapefileInput = document.getElementById('msShapefile');
        const files = msShapefileInput.files;
        if (files.length > 0) {
            uploadShapefile(files, 'ms');
        } else {
            console.error('No MS shapefile files provided.');
        }
    });

    
    function uploadShapefile(files, type) {
        console.log('uploadShapefile called with type:', type);
        if (files.length === 0) {
            console.error(`No ${type} shapefile files provided.`);
            const statusElement = document.getElementById(`${type}ShapefileUploadStatus`);
            statusElement.innerHTML = 'Error: No files provided';
            return;
        }
        
        const endpoint = type === 'rgb' ? '/geoimaging/upload_rgb_shapefile' : '/geoimaging/upload_ms_shapefile';
        const formData = new FormData();
        
        // Convert the FileList to an array
        const fileArray = Array.from(files);

        for (const file of fileArray) {
            formData.append(`${type}_shapefile`, file);
        }
        
        if (type === 'rgb') {
            uploadedRGBShapefileFile = fileArray[0]; // Store the first file from the array
        } else if (type === 'ms') {
            uploadedMSShapefileFile = fileArray[0]; // Store the first file from the array
        }

        fetch(endpoint, {
            method: 'POST',
            body: formData
        })
        .then(response => {
            console.log('Server response:', response);
            if (!response.ok) {
                throw new Error(`HTTP error ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            const statusElement = document.getElementById(`${type}ShapefileUploadStatus`);
            if (data.success) {
                statusElement.innerHTML = '&#10004;'; // Checkmark
                const columnNames = data.column_names;
                const polygonIds = data.polygon_ids;
                const geometries = data.geometries;
    
                const polygonIdsDropdown = document.getElementById(`${type}PolygonIdsDropdown`);
                const geometriesDropdown = document.getElementById(`${type}GeometriesDropdown`);
    
                // Clear the existing options
                polygonIdsDropdown.innerHTML = '';
                geometriesDropdown.innerHTML = '';
    
                // Add an option for column names
                const columnNamesOption = document.createElement('option');
                columnNamesOption.value = '';
                columnNamesOption.text = 'Select Column Name';
                polygonIdsDropdown.add(columnNamesOption);
                geometriesDropdown.add(columnNamesOption.cloneNode(true));
    
                // Populate the column names in both dropdowns
                columnNames.forEach(columnName => {
                    const option = document.createElement('option');
                    option.value = columnName;
                    option.text = columnName;
                    polygonIdsDropdown.add(option);
                    geometriesDropdown.add(option.cloneNode(true));
                });
            } else {
                statusElement.innerHTML = 'Error: ' + data.error;
                console.error(`Error processing ${type} shapefile:`, data.error);
            }
        })
        .catch(error => {
            const statusElement = document.getElementById(`${type}ShapefileUploadStatus`);
            statusElement.innerHTML = 'Error: ' + error;
            console.error('Error communicating with the server:', error);
        });
    }

// CREATE DATASETS -----------------------------------------------------------------------------------------
    // Event listener for creating the RGB2Dataset
    const rgbCreateDatasetBtn = document.getElementById('rgbCreateDatasetBtn');
    rgbCreateDatasetBtn.addEventListener('click', createRGB2Dataset);

    // Event listener for creating the MS2Dataset
    const msCreateDatasetBtn = document.getElementById('msCreateDatasetBtn');
    msCreateDatasetBtn.addEventListener('click', createMS2Dataset);

    // Function to create the RGB2Dataset
    function createRGB2Dataset() {
        const rgbShapefileInput = document.getElementById('rgbShapefile');
        const rgbShapefileFiles = rgbShapefileInput.files;

        if (uploadedRGBImageFile && rgbShapefileFiles.length > 0) {
            const formData = new FormData();
            formData.append('imageType', 'rgb');
            formData.append('rgbImageInput', uploadedRGBImageFile);

            // Append each shapefile file to the FormData
            for (const file of rgbShapefileFiles) {
                formData.append('rgbShapefile', file);
            }

            $.ajax({
                url: '/geoimaging/create_dataset',
                type: 'POST',
                data: formData,
                contentType: false,
                processData: false,
                success: function(response) {
                    // Handle the successful creation of the RGB2Dataset
                    const statusElement = document.getElementById('rgbDatasetStatus');
                    if (response.crs_mismatch) {
                        handleCRSMismatch(response);
                    } else if (response.success) {
                        if (response.crs_matched) {
                            crsMismatchMessage.textContent = 'CRS is matched for RGB dataset.';
                            crsMismatchSuccessMessage.textContent = '';
                            $(crsMismatchModal).modal('show');
                        }
                        statusElement.innerHTML = '&#10004;'; // Checkmark
                    } else {
                        statusElement.innerHTML = 'Error: ' + response.error;
                    }
                },
                error: function(xhr, status, error) {
                    // Handle the error case
                    const statusElement = document.getElementById('rgbDatasetStatus');
                    statusElement.innerHTML = 'Error: ' + error;
                }
            });
        } else {
            console.error('Missing required files for creating RGB2Dataset');
        }
    }

    // Function to create the MS2Dataset
    function createMS2Dataset() {
        const msShapefileInput = document.getElementById('msShapefile');
        const msShapefileFiles = msShapefileInput.files;

        if (
            uploadedRBandFile &&
            uploadedGBandFile &&
            uploadedBBandFile &&
            uploadedREBandFile &&
            uploadedNIRBandFile &&
            msShapefileFiles.length > 0
        ) {
            const formData = new FormData();
            formData.append('imageType', 'ms');
            formData.append('r_band', uploadedRBandFile);
            formData.append('g_band', uploadedGBandFile);
            formData.append('b_band', uploadedBBandFile);
            formData.append('re_band', uploadedREBandFile);
            formData.append('nir_band', uploadedNIRBandFile);

            // Append each shapefile file to the FormData
            for (const file of msShapefileFiles) {
                formData.append('msShapefile', file);
            }

            $.ajax({
                url: '/geoimaging/create_dataset',
                type: 'POST',
                data: formData,
                contentType: false,
                processData: false,
                success: function(response) {
                    // Handle the successful creation of the MS2Dataset
                    const statusElement = document.getElementById('msDatasetStatus');
                    if (response.crs_mismatch) {
                        handleCRSMismatch(response);
                    } else if (response.success) {
                        if (response.crs_matched) {
                            crsMismatchMessage.textContent = 'CRS is matched for MS dataset.';
                            crsMismatchSuccessMessage.textContent = '';
                            $(crsMismatchModal).modal('show');
                        } else {
                            statusElement.innerHTML = '&#10004;'; // Checkmark
                        }
                    }
                    statusElement.innerHTML = 'Error: ' + response.error;
                },
                error: function(xhr, status, error) {
                    // Handle the error case
                    const statusElement = document.getElementById('msDatasetStatus');
                    statusElement.innerHTML = 'Error: ' + error;
                }
            });
        } else {
            console.error('Missing required files for creating MS2Dataset');
        }
    }

// REPROJECT THE CRS -----------------------------------------------------------------------------------------
    const crsMismatchModal = document.getElementById('crsMismatchModal');
    const crsMismatchMessage = document.getElementById('crsMismatchMessage');
    const crsMismatchSuccessMessage = document.getElementById('crsMismatchSuccessMessage');
    const reprojectBtn = document.getElementById('reprojectBtn');

    function handleCRSMismatch(response) {
        const crsMismatchFooter = document.querySelector('.modal-footer');
        const crsMismatchDiv = crsMismatchFooter.querySelector('.crs-mismatch');
        const crsMatchedDiv = crsMismatchFooter.querySelector('.crs-matched');

        if (response.crs_mismatch) {
            crsMismatchMessage.textContent = response.crs_mismatch;
            crsMismatchSuccessMessage.textContent = '';
            crsMismatchDiv.style.display = 'block';
            crsMatchedDiv.style.display = 'none';
            $(crsMismatchModal).modal('show');
    
            reprojectBtn.addEventListener('click', function() {
                const imageType = document.querySelector('input[name="imageType"]:checked').value;
    
                fetch('/geoimaging/reproject_shapefile', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ imageType })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        // Shapefile reprojected successfully
                        crsMismatchSuccessMessage.textContent = data.message; // Display the success message
                        // Proceed with dataset creation
                        if (imageType === 'rgb') {
                            createRGBDataset();
                        } else {
                            createMSDataset();
                        }
                    } else {
                        console.error('Error reprojecting shapefile:', data.error);
                    }
                })
                .catch(error => {
                    console.error('Error communicating with the server:', error);
                });
            });
        } else {
            crsMismatchMessage.textContent = 'CRS is matched.';
            crsMismatchSuccessMessage.textContent = '';
            crsMismatchDiv.style.display = 'none';
            crsMatchedDiv.style.display = 'block';
            $(crsMismatchModal).modal('show');
        }
    }      

// SENDING AND FETCHING FILES --------------------------------------------------------------------------------------------
    function fetchRGBOutputFiles() {
        const outputFilesContainer = document.getElementById('rgbOutputFilesContainer');
        const fileList = outputFilesContainer.querySelector('ul');

        fetch('/geoimaging/get_rgb_output_files')
            .then(response => response.json())
            .then(data => {
                fileList.innerHTML = '';

                if (data.files.length === 0) {
                    const noFilesMessage = document.createElement('li');
                    noFilesMessage.textContent = 'No files available';
                    fileList.appendChild(noFilesMessage);
                } else {
                    data.files.forEach(file => {
                        const listItem = document.createElement('li');
                        const downloadLink = document.createElement('a');
                        downloadLink.href = `/download/${file}`;
                        downloadLink.download = file;
                        downloadLink.textContent = file;
                        listItem.appendChild(downloadLink);
                        fileList.appendChild(listItem);
                    });
                }
            })
            .catch(error => {
                console.error('Error fetching output files:', error);
            });
    }

    function fetchMSOutputFiles() {
        const outputFilesContainer = document.getElementById('msOutputFilesContainer');
        const fileList = outputFilesContainer.querySelector('ul');
    
        fetch('/geoimaging/get_ms_output_files')
            .then(response => response.json())
            .then(data => {
                fileList.innerHTML = '';
    
                if (data.files.length === 0) {
                    const noFilesMessage = document.createElement('li');
                    noFilesMessage.textContent = 'No files available';
                    fileList.appendChild(noFilesMessage);
                } else {
                    data.files.forEach(file => {
                        const listItem = document.createElement('li');
                        const downloadLink = document.createElement('a');
                        downloadLink.href = `/download/${file}`;
                        downloadLink.download = file;
                        downloadLink.textContent = file;
                        listItem.appendChild(downloadLink);
                        fileList.appendChild(listItem);
                    });
                }
            })
            .catch(error => {
                console.error('Error fetching output files:', error);
            });
    }
    
    const downloadAllBtn = document.getElementById('downloadAllBtn');
    downloadAllBtn.addEventListener('click', downloadAllFiles);

    function downloadAllFiles() {
        fetch('/geoimaging/get_rgb_output_files')
            .then(response => response.json())
            .then(data => {
                data.files.forEach(file => {
                    const link = document.createElement('a');
                    link.href = `/download/${file}`;
                    link.download = file;
                    link.click();
                });
            })
            .catch(error => {
                console.error('Error fetching output files:', error);
            });
    }

    const downloadAllMSBtn = document.getElementById('downloadAllMSBtn');
    downloadAllMSBtn.addEventListener('click', downloadAllMSFiles);

    function downloadAllMSFiles() {
        fetch('/geoimaging/get_ms_output_files')
            .then(response => response.json())
            .then(data => {
                data.files.forEach(file => {
                    const link = document.createElement('a');
                    link.href = `/download/${file}`;
                    link.download = file;
                    link.click();
                });
            })
            .catch(error => {
                console.error('Error fetching output files:', error);
            });
    }

    const socket = io();

    socket.on('connect', () => {
        console.log('Connected to server');
    });

    socket.on('output_files_updated', (data) => {
        if (data.fileType === 'rgb') {
            fetchRGBOutputFiles();
        } else if (data.fileType === 'ms') {
            fetchMSOutputFiles(); // Implement this function similarly to fetchRGBOutputFiles
        }
    });

// Event listener for fetching output files
    function fetchOutputFiles(fileType) {
        const outputFilesContainer = document.querySelector(`#${fileType}OutputFilesContainer ul`);
        outputFilesContainer.innerHTML = '';
        
        fetch(`/geoimaging/output_files?type=${fileType}`)
            .then(response => response.json())
            .then(data => {
                if (data.files.length === 0) {
                    outputFilesContainer.innerHTML = '<li>No files available</li>';
                } else {
                    data.files.forEach(file => {
                        const listItem = document.createElement('li');
                        const downloadLink = document.createElement('a');
                        downloadLink.href = `/download/${file}`;
                        downloadLink.download = file;
                        downloadLink.textContent = file;
                        listItem.appendChild(downloadLink);
                        outputFilesContainer.appendChild(listItem);
                    });
                }
            })
            .catch(error => {
                console.error('Error fetching output files:', error);
            });
    }

    // Call the fetchOutputFiles function when the page loads
    document.addEventListener('DOMContentLoaded', () => {
        fetchOutputFiles('rgb');
        fetchOutputFiles('ms');
    });
    
    function sendFileToServer(file, endpoint, queryParams = {}) {
        console.log('sendFileToServer called with file:', file);
        const formData = new FormData();
        formData.append('file', file);
    
        const queryString = new URLSearchParams(queryParams).toString();
        const url = `/geoimaging${endpoint}?${queryString}`;
    
        return new Promise((resolve, reject) => {
            fetch(url, {
                method: 'POST',
                body: formData
            })
            .then(response => {
                if (response.ok) {
                    return response.json();
                } else {
                    reject(new Error(`HTTP error ${response.status}`));
                }
            })
            .then(data => {
                resolve(data);
            })
            .catch(error => {
                reject(error);
            });
        });
    }

// CLIP IMAGE FUNCTIONS -------------------------------------------------------------------------
    const rgbClipBtn = document.getElementById('rgbClipBtn');
    const msClipBtn = document.getElementById('msClipBtn');

    rgbClipBtn.addEventListener('click', clipRGBImage);
    msClipBtn.addEventListener('click', clipMSBand);

    function clipRGBImage() {
        if (!uploadedRGBImageFile || !uploadedRGBShapefileFile) {
            console.error('Please provide both the RGB image and shapefile.');
            return;
        }
    
        const formData = new FormData();
        formData.append('rgb_image', uploadedRGBImageFile);
        formData.append('rgb_shapefile', uploadedRGBShapefileFile);
    
        fetch('/geoimaging/clip_rgb_image', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                console.log('RGB image clipped successfully');
                // can perform additional actions after successful clipping
            } else {
                console.error(`Error clipping RGB image:`, data.error);
                // Handle the error case
            }
        })
        .catch(error => {
            console.error('Error communicating with the server:', error);
        });
    }    

    function clipMSBand() {
        if (
            !uploadedRBandFile ||
            !uploadedGBandFile ||
            !uploadedBBandFile ||
            !uploadedREBandFile ||
            !uploadedNIRBandFile ||
            !uploadedMSShapefileFile
        ) {
            console.error('Please provide all the required MS band files and shapefile.');
            return;
        }
    
        const formData = new FormData();
        formData.append('r_band', uploadedRBandFile);
        formData.append('g_band', uploadedGBandFile);
        formData.append('b_band', uploadedBBandFile);
        formData.append('re_band', uploadedREBandFile);
        formData.append('nir_band', uploadedNIRBandFile);
        formData.append('ms_shapefile', uploadedMSShapefileFile);
    
        fetch('/geoimaging/clip_ms_band', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                console.log('MS bands clipped successfully');
                // can perform additional actions after successful clipping
            } else {
                console.error(`Error clipping MS bands:`, data.error);
                // Handle the error case
            }
        })
        .catch(error => {
            console.error('Error communicating with the server:', error);
        });
    }
});