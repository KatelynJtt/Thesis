// hist_cc_shape_visual.js
let maxIndex = 0;
let isHistogramVisualization = false;
let dataType = 'rgb';
let selectedBandShow = 'nir';
let okBtn;
let popup;
let plotIndexInput;

// Define the initialization functions in the global scope
function initHistogramVisualization(type) {
    console.log('initHistogramVisualization called with type:', type);
    dataType = type;
    loadPlotIndexPopup(true);
}

function initCCVisualization(type) {
    console.log('initCCVisualization called with type:', type);
    dataType = type;
    if (dataType === 'ms') {
        loadBandSelectionPopup();
    } else {
        loadPlotIndexPopup(false);
    }
}

function initShapefileVisualization(type) {
    console.log('initShapefileVisualization called with type:', type);
    dataType = type;
    if (dataType === 'rgb') {
        visualizeShapefile();
    } else {
        visualizeShapefile(selectedBandShow);
    }
}

function loadPlotIndexPopup(isHistogram) {
    console.log('loadPlotIndexPopup called with isHistogram:', isHistogram);
    isHistogramVisualization = isHistogram;

    // Load the plot index popup template
    fetch('/geoimaging/plot_index_popup')
        .then(response => response.text())
        .then(html => {
            console.log('Plot index popup template loaded');
            plotIndexPopupContainer.innerHTML = html;
            popup = document.getElementById('plotIndexPopup');
            plotIndexInput = document.getElementById('plotIndexInput');
            okBtn = document.getElementById('okBtn'); // Get the okBtn element
            
            // Handle input value change
            plotIndexInput.addEventListener('input', () => {
                const value = parseInt(plotIndexInput.value);
                if (isNaN(value) || value < 0 || value > maxIndex) {
                    plotIndexInput.value = Math.max(0, Math.min(maxIndex, value));
                }
            });
            
            if (okBtn) {
                console.log('okBtn element found');
                attachOkButtonListener(); // Attach the event listener
            } else {
                console.error('okBtn element not found in the DOM');
            }
        
            // Fetch the maximum index range from the server
            fetch('/geoimaging/get_max_index_range', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ dataType })
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    console.error('Error:', data.error);
                } else {
                    maxIndex = data.max_index_range;
                    plotIndexInput.max = maxIndex;
                    console.log('Maximum index range:', maxIndex);
                }
            })
            .catch(error => {
                console.error('Error:', error);
            });
        })
        .catch(error => {
            console.error('Error loading plot index popup:', error);
        });
}

function attachOkButtonListener() {
    console.log('Attaching event listener to OK button');
    okBtn.addEventListener('click', handleOkButtonClick);
}

// Handle OK button click
function handleOkButtonClick() {
    console.log('OK button clicked');
    const selectedIndex = parseInt(plotIndexInput.value);
    if (!isNaN(selectedIndex) && selectedIndex >= 0 && selectedIndex <= maxIndex) {
        console.log('Selected index:', selectedIndex);
        // Send the selected plot index to the server
        const body = { plot_index: selectedIndex, is_histogram: isHistogramVisualization, dataType };
        if (dataType === 'ms' && !isHistogramVisualization) {
            body.selectedBandShow = selectedBandShow;
        }

        fetch('/geoimaging/handle_plot_index', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(body)
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                console.error('Error:', data.error);
            } else {
                const binaryString = window.atob(data.image_data);
                const bytes = new Uint8Array(binaryString.length);
                for (let i = 0; i < binaryString.length; i++) {
                    bytes[i] = binaryString.charCodeAt(i);
                }
                const imageBlob = new Blob([bytes], { type: 'image/png' });
                const imageUrl = URL.createObjectURL(imageBlob);
                const imageContainer = document.getElementById('imageContainer');
                const img = new Image();
                img.src = imageUrl;
                imageContainer.innerHTML = '';
                imageContainer.appendChild(img);

                // Close the popup
                popup.style.display = 'none';
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
    } else {
        console.log('Invalid index value:', selectedIndex);
    }
}

function loadBandSelectionPopup() {
    // Load the band selection popup template
    fetch('/geoimaging/band_selection_popup')
        .then(response => response.text())
        .then(html => {
            bandSelectionPopupContainer.innerHTML = html;
            const selectBandBtn = document.getElementById('selectBandBtn');
            const bandSelect = document.getElementById('bandSelect');

            selectBandBtn.addEventListener('click', () => {
                selectedBandShow = bandSelect.value;
                loadPlotIndexPopup(false);
                bandSelectionPopupContainer.innerHTML = '';
            });
        })
        .catch(error => {
            console.error('Error loading band selection popup:', error);
        });
}

function visualizeShapefile(selectedBand = null) {
    const body = { dataType };
    if (selectedBand) {
        body.selectedBandShow = selectedBand;
    }

    fetch('/geoimaging/visualize_shapefile', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(body)
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            console.error('Error:', data.error);
        } else {
            // Handle the server response
            console.log(data.message);
            // Display the image data or perform any other desired action
            const imageContainer = document.getElementById('imageContainer');
            const img = new Image();
            img.src = `data:image/png;base64,${data.image_data}`;
            imageContainer.innerHTML = '';
            imageContainer.appendChild(img);
        }
    })
    .catch(error => {
        console.error('Error:', error);
    });
}


