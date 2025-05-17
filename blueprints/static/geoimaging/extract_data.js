/**
 * This code handles the functionality for the "Extract Dataset Auto" and "Extract Dataset Manual" buttons on the web page.
 * It supports both RGB and Multispectral (MS) datasets.
 */

let selectedPlotIndex = null;

document.addEventListener('DOMContentLoaded', function () {
    // Event listener for the "Extract Dataset Auto" button
    document.getElementById('extractAutoBtn').addEventListener('click', function () {
        const datasetType = this.dataset.type; // Get the dataset type (rgb or ms) from the button's data-type attribute

        // Create a new FormData object to send data to the server
        const formData = new FormData();
        formData.append('datasetType', datasetType);

        // Send a POST request to the '/extract_auto' endpoint
        fetch('/geoimaging/extract_auto', {
            method: 'POST',
            body: formData
        })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    console.log(`${datasetType.toUpperCase()} dataset extracted successfully`);
                    // Perform additional actions after successful extraction, if needed
                } else {
                    console.error(`Error extracting ${datasetType.toUpperCase()} dataset:`, data.error);
                    // Handle the error case, e.g., display an error message to the user
                }
            })
            .catch(error => {
                console.error('Error communicating with the server:', error);
                // Handle network or other communication errors
            });
    });

    // Open the modal when the "Extract Dataset Manual" button is clicked
    const extractManualBtns = document.querySelectorAll('.extractManualBtn');
    extractManualBtns.forEach(btn => {
        btn.addEventListener('click', function () {
            const datasetType = this.dataset.type; // Get the dataset type (rgb or ms) from the button's data-type attribute
            loadModal(datasetType);
        });
    });

    function loadModal(datasetType) {
        const modalHtmlPath = `/static/${datasetType}_extract_manual.html`;

        fetch(modalHtmlPath)
            .then(response => response.text())
            .then(html => {
                const popupContainer = document.getElementById('popupContainer');
                popupContainer.innerHTML = '';
                popupContainer.innerHTML = html;

                const manualExtractionModal = document.getElementById(`${datasetType}ManualExtractionModal`);

                // Add the event listener to the modal
                manualExtractionModal.addEventListener('shown.bs.modal', function () {
                    initializeModal(datasetType);
                });

                $(manualExtractionModal).modal('show');
            })
            .catch(error => {
                console.error('Error loading modal:', error);
            });
    }

    function initializeModal(datasetType) {
        // Get references to the modal elements
        const bandSelect = document.getElementById('bandSelect');
        const prevPageBtn = document.getElementById('prevPageBtn');
        const nextPageBtn = document.getElementById('nextPageBtn');
        const okBtn = document.getElementById('okBtn');
        const showHistogramBtn = document.getElementById('showHistogramBtn');
        const showColorCompositeBtn = document.getElementById('showColorCompositeBtn');

        // Define the showHistogram function
        function showHistogram(plotIndex) {
            if (selectedPlotIndex !== null) {
                const selectedBand = bandSelect.value;
                const threshold = document.getElementById('plotDataTableBody').rows[plotIndex].cells[2].querySelector('input[type="range"]').value;
                const selectedBandShow = document.getElementById('bandShowSelect').value;

                const formData = new FormData();
                formData.append('band', selectedBand);
                formData.append('plot_index', plotIndex);
                formData.append('hist_or_cc', 'true');
                formData.append('selected_band_show', selectedBandShow);

                fetch(`/geoimaging/get_${datasetType}_histogram_data`, {
                    method: 'POST',
                    body: formData
                })
                    .then(response => response.json())
                    .then(data => {
                        if (data.error) {
                            console.error('Error:', data.error);
                        } else {
                            const imageData = data.image_data;

                            const img = new Image();
                            img.src = `data:image/png;base64,${imageData}`;

                            const popupWindow = document.getElementById('popupWindow');
                            popupWindow.innerHTML = '';
                            popupWindow.appendChild(img);
                            popupWindow.style.display = 'block';
                        }
                    })
                    .catch(error => {
                        console.error('Error communicating with the server:', error);
                    });
            } else {
                console.error('No plot selected');
            }
        }

        // Define the showColorComposite function
        function showColorComposite(plotIndex) {
            if (selectedPlotIndex !== null) {
                const selectedBand = bandSelect.value;
                const threshold = document.getElementById('plotDataTableBody').rows[plotIndex].cells[2].querySelector('input[type="range"]').value;
                const selectedBandShow = document.getElementById('bandShowSelect').value;

                const formData = new FormData();
                formData.append('band', selectedBand);
                formData.append('plot_index', plotIndex);
                formData.append('hist_or_cc', 'false');
                formData.append('selected_band_show', selectedBandShow);

                fetch(`/geoimaging/get_${datasetType}_histogram_data`, {
                    method: 'POST',
                    body: formData
                })
                    .then(response => response.json())
                    .then(data => {
                        if (data.error) {
                            console.error('Error:', data.error);
                        } else {
                            const imageData = data.image_data;

                            const img = new Image();
                            img.src = `data:image/png;base64,${imageData}`;

                            const popupWindow = document.getElementById('popupWindow');
                            popupWindow.innerHTML = '';
                            popupWindow.appendChild(img);
                            popupWindow.style.display = 'block';
                        }
                    })
                    .catch(error => {
                        console.error('Error communicating with the server:', error);
                    });
            } else {
                console.error('No plot selected');
            }
        }

        // Add event listeners or call the functions as needed
        showHistogramBtn.addEventListener('click', showHistogram);
        showColorCompositeBtn.addEventListener('click', showColorComposite);

        // Fetch the initial data (list of bands and plot data for the first page)
        function fetchInitialData() {
            fetch(`/geoimaging/${datasetType}_manual_extraction`)
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        console.error('Error:', data.error);
                    } else {
                        populateBandDropdown(data.bands);
                        renderPlotData(data.plotData, 1);
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                });
        }

        // Populate the band dropdown
        function populateBandDropdown(bands) {
            bandSelect.innerHTML = '';
            bands.forEach(band => {
                const option = document.createElement('option');
                option.value = band;
                option.text = band;
                bandSelect.add(option);
            });
        }

        // Render the plot data in the table
        function renderPlotData(plotData, currentPage) {
            const tableBody = document.getElementById('plotDataTableBody');
            tableBody.innerHTML = '';
            plotData.forEach((plot, index) => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${plot.plot_indx}</td>
                    <td>${plot.plot_name}</td>
                    <td><input type="range" min="${plot.min_th}" max="${plot.max_th}" value="${plot.threshold}"></td>
                    <td><button class="btn btn-primary" onclick="showHistogram(${index})">Histogram</button></td>
                    <td><button class="btn btn-primary" onclick="showColorComposite(${index})">Color Composite</button></td>
                `;
                tableBody.appendChild(row);
            });
            document.getElementById('currentPageDisplay').textContent = `Page ${currentPage}`;
        }

        // Handle band dropdown change
        bandSelect.addEventListener('change', function () {
            const selectedBand = this.value;
            fetchPlotData(selectedBand, 1);
        });

        // Handle pagination buttons
        prevPageBtn.addEventListener('click', function () {
            const currentPage = parseInt(document.getElementById('currentPageDisplay').textContent.split(' ')[1]);
            if (currentPage > 1) {
                const selectedBand = bandSelect.value;
                fetchPlotData(selectedBand, currentPage - 1);
            }
        });

        nextPageBtn.addEventListener('click', function () {
            const currentPage = parseInt(document.getElementById('currentPageDisplay').textContent.split(' ')[1]);
            const selectedBand = bandSelect.value;
            fetchPlotData(selectedBand, currentPage + 1);
        });

        // Fetch plot data for the selected band and page
        function fetchPlotData(band, page) {
            const url = `/geoimaging/${datasetType}_get_plot_data?band=${band}&page=${page}`;
            fetch(url)
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        console.error('Error:', data.error);
                    } else {
                        renderPlotData(data.plotData, data.currentPage);
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                });
        }

        // Handle "OK" button click
        okBtn.addEventListener('click', function () {
            const selectedBand = bandSelect.value;
            const thresholds = Array.from(document.getElementById('plotDataTableBody').rows).map(row => row.cells[2].querySelector('input[type="range"]').value);

            const formData = new FormData();
            formData.append('band', selectedBand);
            thresholds.forEach(threshold => {
                formData.append('thresholds[]', threshold);
            });

            fetch(`/geoimaging/${datasetType}_manual_extraction_ok`, {
                method: 'POST',
                body: formData
            })
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        console.error('Error:', data.error);
                    } else {
                        alert(data.message);
                        // Optionally additional actions after successful manual extraction
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                });
        });

        // Fetch the initial data
        fetchInitialData();
    }
});
