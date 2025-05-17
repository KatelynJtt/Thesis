(function() {
    const socket = io('/geoimaging');

    socket.on('connect', () => {
        console.log('Connected to server');
    });

    socket.on('output_files_updated', (data) => {
        if (data.fileType === 'rgb') {
            fetchRGBOutputFiles();
        } else if (data.fileType === 'ms') {
            fetchMSOutputFiles();
        }
    });

    socket.on('clipping_complete', function(data) {
        if (data.type === 'rgb') {
            document.getElementById('rgbPreviewBtn').disabled = false;
            const statusDiv = document.getElementById('rgbClipStatus');
            statusDiv.textContent = 'Clipping completed successfully';
            statusDiv.className = 'alert alert-success';
        } else if (data.type === 'ms') {
            document.getElementById('msPreviewBtn').disabled = false;
            const statusDiv = document.getElementById('msClipStatus');
            statusDiv.textContent = 'Clipping completed successfully';
            statusDiv.className = 'alert alert-success';
        }
    });
    
    function setupEventListeners() {
        console.log('Setting up event listeners');
        // Image type radio buttons
        const imageTypeRadios = document.querySelectorAll('input[name="imageType"]');
        imageTypeRadios.forEach(radio => {
            radio.addEventListener('change', toggleLayout);
        });

        // Check which tab / radio button is checked
        const isMS = document.getElementById('msImageRadio').checked;

        // Tab toggling for RGB Extract
        $('#rgbExtractTabs a').on('click', function (e) {
            e.preventDefault();
            $(this).tab('show');
        });

        // Tab toggling for RGB Visualization
        $('#rgbVisualizeTabs a').on('click', function (e) {
            e.preventDefault();
            $(this).tab('show');
            setMaxIndexForInputs('rgb');
        });

        // Tab toggling for MS Extract
        $('#msExtractTabs a').on('click', function (e) {
            e.preventDefault();
            $(this).tab('show');
        });

        // Tab toggling for MS Visualization
        $('#msVisualizeTabs a').on('click', function (e) {
            e.preventDefault();
            $(this).tab('show');
            setMaxIndexForInputs('ms');
        });

        // Initialize first tabs as active
        $('#rgbExtractTabs a:first').tab('show');
        $('#rgbVisualizeTabs a:first').tab('show');
        $('#msExtractTabs a:first').tab('show');
        $('#msVisualizeTabs a:first').tab('show');
    
        if (isMS) {
            // MS Upload Buttons
            ['r', 'g', 'b', 're', 'nir'].forEach(band => {
                const uploadBtn = document.getElementById(`${band}BandUploadBtn`);
                if (uploadBtn) {
                    uploadBtn.addEventListener('click', function() {
                        const fileInput = document.getElementById(`${band}_band`);
                        const files = fileInput.files;
                        if (files.length > 0) {
                            const spinner = document.getElementById(`${band}BandSpinner`);
                            spinner.style.display = 'inline-block';
                            
                            const formData = new FormData();
                            formData.append('file', files[0]);
                            formData.append('band_type', `${band}_band`);
                            
                            fetch('/geoimaging/upload_ms_band', {
                                method: 'POST',
                                body: formData
                            })
                            .then(response => response.json())
                            .then(data => {
                                spinner.style.display = 'none';
                                if (data.success) {
                                    document.getElementById(`${band}BandUploadStatus`).textContent = 'Upload successful';
                                } else {
                                    document.getElementById(`${band}BandUploadStatus`).textContent = data.error || 'Upload failed';
                                }
                            });
                        }
                    });
                }
            });            
    
            const msShapefileBtn = document.getElementById('msShapefileUploadBtn');
            if (msShapefileBtn) {
                msShapefileBtn.addEventListener('click', function() {
                    console.log('MS Shapefile Upload Button Clicked');
                    const fileInput = document.getElementById('msShapefile');
                    const files = fileInput.files;
                    if (files.length > 0) {
                        const spinner = document.getElementById(`msShapefileSpinner`);
                        if (spinner) {
                            spinner.style.display = 'inline-block';
                        }
                        const formData = new FormData();
                        for (let file of files) {
                            formData.append('ms_shapefile', file);
                        }
                       
                        fetch('/geoimaging/upload_ms_shapefile', {
                            method: 'POST',
                            body: formData
                        })
                        .then(response => response.json())
                        .then(data => {
                            if (data.success) {
                                document.getElementById('msShapefileUploadStatus').textContent = 'Upload successful';
                                displayShapefileData(data, 'ms');
                            } else {
                                document.getElementById('msShapefileUploadStatus').textContent = data.error || 'Upload failed';
                            }
                        });
                    }
                });
            }

            // MS Clip Button
            const msClipBtn = document.getElementById('msClipBtn');
            if (msClipBtn) {
                msClipBtn.addEventListener('click', function() {
                    console.log('MS Clip Button Clicked');
                    handleClip('ms');
                });
            }
            // MS Preview Setup
            const msPreviewBtn = document.getElementById('msPreviewBtn');
            const msClearPreviewBtn = document.getElementById('msClearPreviewBtn');
            const msDownloadAllBtn = document.getElementById('msDownloadAllBtn');
            const msPreviewStatus = document.getElementById('msPreviewStatus');
            const msClippedImagesPreview = document.getElementById('msClippedImagesPreview');
            const msImageGrid = document.getElementById('msImageGrid');

            if (msPreviewBtn) {
                msPreviewBtn.addEventListener('click', () => handlePreview('ms'));
            }

            const msCollapseBtn = document.getElementById('msCollapseBtn');
            if (msCollapseBtn) {
                msCollapseBtn.addEventListener('click', () => {
                    const previewContainer = document.getElementById('msClippedImagesPreview');
                    previewContainer.classList.toggle('show');
                    const icon = msCollapseBtn.querySelector('i');
                    icon.classList.toggle('fa-chevron-down');
                    icon.classList.toggle('fa-chevron-up');
                    document.getElementById('msPreviewStatus').style.display = 'none';
                    document.getElementById('msClipStatus').style.display = 'none';
                });
            }

            if (msClearPreviewBtn) {
                msClearPreviewBtn.disabled = true;
                msClearPreviewBtn.addEventListener('click', () => {
                    console.log('MS Clear Preview Button Clicked');
                    msImageGrid.innerHTML = '';
                    msClippedImagesPreview.classList.remove('show');
                    msPreviewStatus.style.display = 'none';
                    // Don't disable the preview button
                    msPreviewBtn.disabled = false;
                    // Reset other buttons to initial state
                    msClearPreviewBtn.disabled = true;
                    msDownloadAllBtn.disabled = true;
                    msCollapseBtn.disabled = true;
                });
            }

            if (msDownloadAllBtn) {
                msDownloadAllBtn.disabled = true;
                msDownloadAllBtn.addEventListener('click', () => {
                    console.log('MS Download All Button Clicked');
                    const images = JSON.parse(msClippedImagesPreview.dataset.images || '[]');
                    images.forEach(image => {
                        const link = document.createElement('a');
                        link.href = `data:image/png;base64,${image.data}`;
                        link.download = `ms_plot_${image.plot_id}.png`;
                        document.body.appendChild(link);
                        link.click();
                        link.remove();
                    });
                    msPreviewStatus.style.display = 'none';
                });
            }

            // MS Auto Data Extract
            const msAutoExtractBtn = document.getElementById('msAutoExtractBtn');
            if (msAutoExtractBtn) {
                msAutoExtractBtn.addEventListener('click', () => 
                    extractDataset('ms', 'auto')
                );
            }

            // MS Manual Data Extract
            const msManualConfirmBtn = document.getElementById('msManualConfirmBtn');
            if (msManualConfirmBtn) {
                msManualConfirmBtn.addEventListener('click', () => {
                    const selectedVI = document.getElementById('msSelectedVI').value;
                    const thresholds = Array.from(document.querySelectorAll('.ms-threshold-input'))
                        .map(input => input.value);
                    
                    extractDataset('ms', 'manual', {
                        selectedVI: selectedVI,
                        thresholds: thresholds
                    });
                });
            }

            // MS Histogram Visualization
            $('#msHistVisualizeBtn').on('click', function() {
                const plotIndex = $('#msHistPlotIndex').val();
                console.log('Plotting histogram for MS band:', plotIndex);
                visualizeHistogram('ms', plotIndex);
            });
            // MS CC Visualization
            $('#msCCVisualizeBtn').on('click', function() {
                const plotIndex = $('#msCCPlotIndex').val();
                console.log('Plotting CC for MS band:', plotIndex);
                visualizeCC('ms', plotIndex);
            });
            // MS Shapefile Visualization
            $('#msShapefileVisualizeBtn').on('click', function() {
                console.log('Visualizing MS Shapefile');
                visualizeShapefile('ms');
            });
    
        } else {
            // RGB Upload Buttons
            const rgbImageUploadBtn = document.getElementById('rgbImageUploadBtn');
            if (rgbImageUploadBtn) {
                rgbImageUploadBtn.addEventListener('click', function() {
                    console.log('RGB Image Upload Button Clicked');
                    const fileInput = document.getElementById('rgbImageInput');
                    const file = fileInput.files[0];
                    if (file) {
                        const formData = new FormData();
                        formData.append('file', file);

                        const spinner = document.getElementById('rgbImageSpinner');
                        if (spinner) {
                            spinner.style.display = 'inline-block';
                        }
                        
                        fetch('/geoimaging/upload_rgb_image', {
                            method: 'POST',
                            body: formData
                        })
                        .then(response => response.json())
                        .then(data => {
                            spinner.style.display = 'none';
                            if (data.success) {
                                document.getElementById('rgbImageUploadStatus').textContent = 'Upload successful';
                            } else {
                                document.getElementById('rgbImageUploadStatus').textContent = data.error || 'Upload failed';
                            }
                        });
                    }
                });
            }
    
            const rgbShapefileBtn = document.getElementById('rgbShapefileUploadBtn');
            if (rgbShapefileBtn) {
                rgbShapefileBtn.addEventListener('click', function() {
                    console.log('RGB Shapefile Upload Button Clicked');
                    const fileInput = document.getElementById('rgbShapefile');
                    const files = fileInput.files;
                    if (files.length > 0) {
                        const spinner = document.getElementById('rgbShapefileSpinner');
                        if (spinner) {
                            spinner.style.display = 'inline-block';
                        }
                        const formData = new FormData();
                        for (let file of files) {
                            formData.append('rgb_shapefile', file);
                        }
                       
                        fetch('/geoimaging/upload_rgb_shapefile', {
                            method: 'POST',
                            body: formData
                        })
                        .then(response => response.json())
                        .then(data => {
                            if (spinner) {
                                spinner.style.display = 'none';  // Hide spinner after response
                            }
                            if (data.success) {
                                document.getElementById('rgbShapefileUploadStatus').textContent = 'Upload successful';
                                displayShapefileData(data, 'rgb');
                            } else {
                                document.getElementById('rgbShapefileUploadStatus').textContent = data.error || 'Upload failed';
                            }
                        });
                    }
                });
            }

            // RGB Clip Button
            const rgbClipBtn = document.getElementById('rgbClipBtn');
            if (rgbClipBtn) {
                rgbClipBtn.addEventListener('click', function() {
                    console.log('RGB Clip Button Clicked');
                    handleClip('rgb');
                });
            }

            // RGB Preview Setup
            const rgbPreviewBtn = document.getElementById('rgbPreviewBtn');
            const rgbClearPreviewBtn = document.getElementById('rgbClearPreviewBtn');
            const rgbDownloadAllBtn = document.getElementById('rgbDownloadAllBtn');
            const rgbPreviewStatus = document.getElementById('rgbPreviewStatus');
            const rgbClippedImagesPreview = document.getElementById('rgbClippedImagesPreview');
            const rgbImageGrid = document.getElementById('rgbImageGrid');

            if (rgbPreviewBtn) {
                rgbPreviewBtn.addEventListener('click', () => handlePreview('rgb'));
            }

            const rgbCollapseBtn = document.getElementById('rgbCollapseBtn');
            if (rgbCollapseBtn) {
                rgbCollapseBtn.addEventListener('click', () => {
                    const previewContainer = document.getElementById('rgbClippedImagesPreview');
                    previewContainer.classList.toggle('show');
                    const icon = rgbCollapseBtn.querySelector('i');
                    icon.classList.toggle('fa-chevron-down');
                    icon.classList.toggle('fa-chevron-up');
                    document.getElementById('rgbPreviewStatus').style.display = 'none';
                    document.getElementById('rgbClipStatus').style.display = 'none';
                });
            }

            if (rgbClearPreviewBtn) {
                rgbClearPreviewBtn.disabled = true;
                rgbClearPreviewBtn.addEventListener('click', () => {
                    console.log('RGB Clear Preview Button Clicked');
                    rgbImageGrid.innerHTML = '';
                    rgbClippedImagesPreview.classList.remove('show');
                    rgbPreviewStatus.style.display = 'none';
                    // Don't disable the preview button
                    rgbPreviewBtn.disabled = false;
                    // Reset other buttons to initial state
                    rgbClearPreviewBtn.disabled = true;
                    rgbDownloadAllBtn.disabled = true;
                    rgbCollapseBtn.disabled = true;
                });
            }

            if (rgbDownloadAllBtn) {
                rgbDownloadAllBtn.disabled = true;
                rgbDownloadAllBtn.addEventListener('click', () => {
                    console.log('RGB Download All Button Clicked');
                    const images = JSON.parse(rgbClippedImagesPreview.dataset.images || '[]');
                    images.forEach(image => {
                        const link = document.createElement('a');
                        link.href = `data:image/png;base64,${image.data}`;
                        link.download = `rgb_plot_${image.plot_id}.png`;
                        document.body.appendChild(link);
                        link.click();
                        link.remove();
                    });
                    rgbPreviewStatus.style.display = 'none';
                });
            }

            // RGB Auto Data Extraction
            const rgbAutoExtractBtn = document.getElementById('rgbAutoExtractBtn');
            if (rgbAutoExtractBtn) {
                rgbAutoExtractBtn.addEventListener('click', () => 
                    extractDataset('rgb', 'auto')
                );
            }

            // RGB Manual Data Extraction
            const rgbManualConfirmBtn = document.getElementById('rgbManualConfirmBtn');
            if (rgbManualConfirmBtn) {
                rgbManualConfirmBtn.addEventListener('click', () => {
                    const selectedVI = document.getElementById('rgbSelectedVI').value;
                    const thresholds = Array.from(document.querySelectorAll('.rgb-threshold-input'))
                        .map(input => input.value);
                    
                    extractDataset('rgb', 'manual', {
                        selectedVI: selectedVI,
                        thresholds: thresholds
                    });
                });
            }

            // RGB Histogram Visualization
            $('#rgbHistVisualizeBtn').on('click', function() {
                const plotIndex = $('#rgbHistPlotIndex').val();
                console.log('RGB Hist Plot Index:', plotIndex);
                visualizeHistogram('rgb', plotIndex);
            });
            // RGB CC Visualization
            $('#rgbCCVisualizeBtn').on('click', function() {
                const plotIndex = $('#rgbCCPlotIndex').val();
                console.log('RGB CC Plot Index:', plotIndex);
                visualizeCC('rgb', plotIndex);
            });
            // RGB Shapefile Visualization
            $('#rgbShapefileVisualizeBtn').on('click', function() {
                console.log('RGB Shapefile Visualization Button Clicked');
                visualizeShapefile('rgb');
            });
        }
    }
    
    function loadGeoimagingStyles() {
        if (!document.getElementById('geoimaging-bootstrap')) {
            const link = document.createElement('link');
            link.id = 'geoimaging-bootstrap';
            link.rel = 'stylesheet';
            link.href = 'https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css';
            document.head.appendChild(link);
        }
    }

    // Update the initialization function
    window.initializeGeoimaging = function() {
        // Only initialize if we're on the geoimaging tab
        const geoimagingApp = document.getElementById('geoimaging_app');
        if (!geoimagingApp) {
            return; // Exit if not on geoimaging tab
        }

        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', initializeApp);
        } else {
            initializeApp();
        }
    }
    
    function initializeApp() {
        console.log('Geoimaging script loaded');
        //loadGeoimagingStyles(); // Add this line to load Bootstrap styles
        const appDiv = document.getElementById('geoimaging_app');
        setupEventListeners();
        
        // Check URL parameters and set initial layout
        const urlParams = new URLSearchParams(window.location.search);
        const imageType = urlParams.get('imageType');
    
        if (imageType === 'ms') {
            document.getElementById('msImageRadio').checked = true;
            document.getElementById('rgbLayout').style.display = 'none';
            document.getElementById('msLayout').style.display = 'block';
        } else {
            document.getElementById('rgbImageRadio').checked = true;
            document.getElementById('rgbLayout').style.display = 'block';
            document.getElementById('msLayout').style.display = 'none';
        }

        // Set up tooltip for Clipping
        var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
        var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl)
        })

    }

    // Call initialization when DOM is ready
    initializeGeoimaging();

    function toggleLayout() {
        if (this.value === 'ms') {
            document.getElementById('rgbLayout').style.display = 'none';
            document.getElementById('msLayout').style.display = 'block';
            // Update URL without page reload
            history.pushState({}, '', '/?imageType=ms');
            setupEventListeners();  // Re-setup listeners for MS layout
        } else {
            document.getElementById('rgbLayout').style.display = 'block';
            document.getElementById('msLayout').style.display = 'none';
            // Update URL without page reload
            history.pushState({}, '', '/');
            setupEventListeners();  // Re-setup listeners for MS layout
        }
    }

    function displayShapefileData(response, dataType) {
        // Check if elements exist based on current view
        const plotIdList = document.getElementById(`${dataType}PlotIdsList`);
        const geometryList = document.getElementById(`${dataType}GeometriesList`);
        
        if (plotIdList && geometryList && response.polygon_ids && response.geometries) {
            plotIdList.innerHTML = response.polygon_ids.join('<br>');
            geometryList.innerHTML = response.geometries.join('<br>');
        }
    }

    function setMaxIndexForInputs(dataType) {
        fetch('/geoimaging/get_max_index_range', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                dataType: dataType  // Ensure the property name matches exactly
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.max_index_range !== undefined) {
                const maxIndex = data.max_index_range;
                if (dataType === 'rgb') {
                    document.getElementById('rgbHistPlotIndex').max = maxIndex;
                    document.getElementById('rgbCCPlotIndex').max = maxIndex;
                } else if (dataType === 'ms') {
                    document.getElementById('msHistPlotIndex').max = maxIndex;
                    document.getElementById('msCCPlotIndex').max = maxIndex;
                }
            }
        })
        .catch(error => {
            console.log('Error fetching max index range:', error);
        });
    }
    
    function handleClip(imageType) {
        const statusDiv = document.getElementById(`${imageType}ClipStatus`);
        const previewBtn = document.getElementById(`${imageType}PreviewBtn`);
        previewBtn.disabled = true;
        
        statusDiv.textContent = 'Checking files...';
        statusDiv.className = 'alert alert-info';
    
        if (imageType === 'rgb') {
            fetch('/geoimaging/check_files')
                .then(response => response.json())
                .then(data => {
                    if (!data.hasImage || !data.hasShapefile) {
                        statusDiv.textContent = 'Please upload both RGB image and shapefile first';
                        statusDiv.className = 'alert alert-warning';
                        return;
                    }
                    
                    statusDiv.textContent = 'Initializing dataset and preparing for clipping...';
                    return fetch('/geoimaging/clip_rgb_image', { method: 'POST' });
                })
                .then(response => response?.json())
                .then(data => {
                    if (data?.success) {
                        statusDiv.textContent = data.message;
                        statusDiv.className = 'alert alert-success';
                        previewBtn.disabled = false;
                        
                        if (!data.crs_matched) {
                            showCRSMismatchDialog(imageType);
                        }
                    } else if (data?.error) {
                        statusDiv.textContent = data.error;
                        statusDiv.className = 'alert alert-danger';
                    }
                });
        } else {
            fetch('/geoimaging/check_files')
                .then(response => response.json())
                .then(data => {
                    if (!data.hasImage || !data.hasShapefile) {
                        statusDiv.textContent = 'Please upload all MS bands and shapefile first';
                        statusDiv.className = 'alert alert-warning';
                        return;
                    }
    
                    statusDiv.textContent = 'Initializing dataset...';
                    
                    return fetch('/geoimaging/clip_ms_band', {
                        method: 'POST'
                    });
                })
                .then(response => response?.json())
                .then(data => {
                    if (data?.success) {
                        statusDiv.textContent = 'Dataset initialized, now clipping bands...';
                        const bands = ['r_band', 'g_band', 'b_band', 're_band', 'nir_band'];
                        let completedBands = 0;
    
                        bands.forEach(band => {
                            const formData = new FormData();
                            formData.append('band_type', band);
    
                            fetch('/geoimaging/clip_ms_band', {
                                method: 'POST',
                                body: formData
                            })
                            .then(response => response.json())
                            .then(data => {
                                if (data.message) {
                                    completedBands++;
                                    statusDiv.textContent = `Processing... ${completedBands}/${bands.length} bands complete`;
                                    
                                    if (completedBands === bands.length) {
                                        statusDiv.textContent = 'Dataset initialized and all bands clipped successfully';
                                        statusDiv.className = 'alert alert-success';
                                        previewBtn.disabled = false;
                                        
                                        if (!data.crs_matched) {
                                            showCRSMismatchDialog(imageType);
                                        }
                                    }
                                } else if (data.error) {
                                    statusDiv.textContent = `Error processing ${band}: ${data.error}`;
                                    statusDiv.className = 'alert alert-danger';
                                }
                            });
                        });
                    } else if (data?.error) {
                        statusDiv.textContent = data.error;
                        statusDiv.className = 'alert alert-danger';
                    }
                });
        }
    }
    
    function handlePreview(imageType) {
        const previewStatus = document.getElementById(`${imageType}PreviewStatus`);
        const imageGrid = document.getElementById(`${imageType}ImageGrid`);
        const previewContainer = document.getElementById(`${imageType}ClippedImagesPreview`);
        const clearBtn = document.getElementById(`${imageType}ClearPreviewBtn`);
        const downloadAllBtn = document.getElementById(`${imageType}DownloadAllBtn`);
        const previewProgress = document.getElementById(`${imageType}PreviewProgress`);
        const progressBar = previewProgress.querySelector('.progress-bar');
    
        // Hide clip status when preview starts
        document.getElementById(`${imageType}ClipStatus`).style.display = 'none';
        // Enable collapse button
        document.getElementById(`${imageType}CollapseBtn`).disabled = false;
    
        previewStatus.textContent = 'Loading previews...';
        previewStatus.className = 'alert alert-info';
    
        // Show progress bar
        previewProgress.style.display = 'block';
        progressBar.style.width = '0%';
        
        // Start progress animation
        let progress = 0;
        const progressInterval = setInterval(() => {
            progress += 5;
            if (progress <= 90) {
                progressBar.style.width = `${progress}%`;
            }
        }, 100);
    
        fetch(`/geoimaging/get_clipped_images?type=${imageType}`)
            .then(response => response.json())
            .then(data => {
                imageGrid.innerHTML = '';
                
                data.images.forEach((image, index) => {
                    const imgContainer = document.createElement('div');
                    imgContainer.className = 'image-container';
                    
                    const img = document.createElement('img');
                    img.src = `data:image/png;base64,${image.data}`;
                    img.alt = `Plot ${image.plot_id}`;
                    img.style.maxWidth = '200px';
                    
                    const downloadBtn = document.createElement('a');
                    downloadBtn.className = 'btn btn-sm btn-secondary mt-2';
                    downloadBtn.innerHTML = 'Download';
                    downloadBtn.href = `data:image/png;base64,${image.data}`;
                    downloadBtn.download = `${imageType}_plot_${image.plot_id}.png`;
                    
                    const label = document.createElement('p');
                    label.textContent = `Plot ${image.plot_id}`;
                    
                    imgContainer.appendChild(img);
                    imgContainer.appendChild(label);
                    imgContainer.appendChild(downloadBtn);
                    imageGrid.appendChild(imgContainer);
    
                    // Update progress based on processed images
                    const currentProgress = Math.min(90 + ((index + 1) / data.images.length * 10), 100);
                    progressBar.style.width = `${currentProgress}%`;
                });
    
                // Complete the progress bar
                clearInterval(progressInterval);
                progressBar.style.width = '100%';
                
                // Hide progress bar after a short delay
                setTimeout(() => {
                    previewProgress.style.display = 'none';
                }, 500);
    
                // Enable the clear and download buttons
                clearBtn.disabled = false;
                downloadAllBtn.disabled = false;
                
                // Show preview and store images data
                previewContainer.classList.add('show');
                previewContainer.dataset.images = JSON.stringify(data.images);
                
                previewStatus.textContent = 'Preview loaded successfully';
                previewStatus.className = 'alert alert-success';
            })
            .catch(error => {
                clearInterval(progressInterval);
                previewProgress.style.display = 'none';
                previewStatus.textContent = 'Error loading previews';
                previewStatus.className = 'alert alert-danger';
            });
    }
        
    function visualizeHistogram(dataType, plotIndex) {
        console.log('Visualizing histogram:', dataType, plotIndex);
        const containerId = dataType === 'rgb' ? 'rgbHistVisualizationContainer' : 'msHistVisualizationContainer';
        let requestBody = {
            dataType: dataType,
            plot_index: parseInt(plotIndex),
            is_histogram: true
        };
    
        if (dataType === 'ms') {
            const selectedBand = document.getElementById('msHistBandSelect').value;
            requestBody.selectedBandShow = selectedBand;
        }
        
        fetch('/geoimaging/handle_plot_index', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody)
        })
        .then(response => response.json())
        .then(data => {
            if (data.image_data) {
                displayVisualization(data.image_data, containerId);
            }
        });
    }
    
    function visualizeCC(dataType, plotIndex) {
        console.log('Visualizing CC:', dataType, plotIndex);
        const containerId = dataType === 'rgb' ? 'rgbCCVisualizationContainer' : 'msCCVisualizationContainer';
        let requestBody = {
            dataType: dataType,
            plot_index: parseInt(plotIndex),
            is_histogram: false
        };
    
        if (dataType === 'ms') {
            const selectedBand = document.getElementById('msCCBandSelect').value;
            requestBody.selectedBandShow = selectedBand;
        }
        
        fetch('/geoimaging/handle_plot_index', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody)
        })
        .then(response => response.json())
        .then(data => {
            if (data.image_data) {
                displayVisualization(data.image_data, containerId);
            }
        });
    }

    function visualizeShapefile(dataType) {
        console.log('Visualizing shapefile:', dataType);
        const containerId = dataType === 'rgb' ? 'rgbShapefileVisualizationContainer' : 'msShapefileVisualizationContainer';
        
        fetch('/geoimaging/visualize_shapefile', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                dataType: dataType
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.image_data) {
                displayVisualization(data.image_data, containerId);
            }
        });
    }
        
    function displayVisualization(imageData, containerId) {
        const container = document.getElementById(containerId);
        const wrapper = document.createElement('div');
        wrapper.className = 'visualization-wrapper';
    
        const img = new Image();
        img.src = 'data:image/png;base64,' + imageData;
        
        const downloadBtn = document.createElement('button');
        downloadBtn.className = 'btn btn-secondary';
        downloadBtn.innerHTML = 'Download';
        downloadBtn.onclick = function() {
            const link = document.createElement('a');
            link.download = `visualization_${Date.now()}.png`;
            link.href = img.src;
            link.click();
        };
    
        wrapper.appendChild(img);
        wrapper.appendChild(downloadBtn);
        container.innerHTML = '';
        container.appendChild(wrapper);
    }

    // Initialize tooltips
    $(function () {
        $('[data-toggle="tooltip"]').tooltip();
    });

    // DATASET EXTRACTION
    function extractDataset(imageType, extractType, additionalData = {}) {
        const formData = new FormData();
        formData.append('imageType', imageType);
        formData.append('extractType', extractType);
        
        // Add any additional data for manual extraction
        Object.entries(additionalData).forEach(([key, value]) => {
            if (Array.isArray(value)) {
                value.forEach(val => formData.append(`${key}[]`, val));
            } else {
                formData.append(key, value);
            }
        });
        
        // Show status
        const statusElement = document.getElementById(`${imageType}DatasetStatus`);
        statusElement.textContent = 'Creating dataset...';
        statusElement.className = 'alert alert-info';
        
        fetch('/geoimaging/create_dataset', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                statusElement.textContent = 'Dataset created successfully';
                statusElement.className = 'alert alert-success';
                
                const modal = $('#datasetCreatedModal');
                modal.data('imageType', imageType);
                modal.data('datasetDir', data.datasetDir);  // Store the directory
                
                $('#saveAndAddBtn').off().on('click', function() {
                    const imageType = $('#datasetCreatedModal').data('imageType');
                    const datasetDir = $('#datasetCreatedModal').data('datasetDir');
                    fetch(`/geoimaging/download_dataset?type=${imageType}&dir=${datasetDir}`)                        .then(response => response.blob())
                        .then(blob => {
                            // Create file for adding to sidebar
                            const file = new File([blob], `${imageType}_dataset_${Date.now()}.xlsx`, {
                                type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                            });
                            window.handleFiles([file]);
                            
                            // Create download link for local save
                            const url = window.URL.createObjectURL(blob);
                            const a = document.createElement('a');
                            a.href = url;
                            a.download = `${imageType}_dataset_${Date.now()}.xlsx`;
                            document.body.appendChild(a);
                            a.click();
                            a.remove();
                            
                            modal.modal('hide');
                        });
                });
                
                $('#saveOnlyBtn').off().on('click', function() {
                    const imageType = $('#datasetCreatedModal').data('imageType');
                    fetch(`/geoimaging/download_dataset?type=${imageType}`)
                        .then(response => response.blob())
                        .then(blob => {
                            const url = window.URL.createObjectURL(blob);
                            const a = document.createElement('a');
                            a.href = url;
                            a.download = `${imageType}_dataset_${Date.now()}.xlsx`;
                            document.body.appendChild(a);
                            a.click();
                            a.remove();
                            modal.modal('hide');
                        });
                });
                                
                modal.modal('show');
            } else {
                statusElement.textContent = data.error || 'Dataset creation failed';
                statusElement.className = 'alert alert-danger';
            }
        });
    }

})();
