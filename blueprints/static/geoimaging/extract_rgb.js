/**
 * This code handles the functionality for the "Extract RGB Dataset Auto" and "Extract Dataset Manual" buttons on the web page.
 * 
 * The "Extract RGB Dataset Auto" button sends a POST request to the '/extract_auto' endpoint to extract the RGB dataset automatically.
 * The "Extract Dataset Manual" button opens a modal window that allows the user to manually extract the RGB dataset.
 * 
 * The manual extraction functionality includes:
 * - Fetching the initial data (list of VIs and plot data for the first page)
 * - Populating the VI dropdown
 * - Rendering the plot data in a table
 * - Handling the VI dropdown change to fetch new plot data
 * - Handling the pagination buttons to navigate through the plot data
 * - Showing the histogram or color composite for a selected plot
 * - Handling the "OK" button click to submit the manual extraction with the selected thresholds
 */
let selectedPlotIndex = null;

document.addEventListener('DOMContentLoaded', function () {

    // Event listener for the "Extract RGB Dataset Auto" button
    document.getElementById('rgbExtractAutoBtn').addEventListener('click', function () {
        // Create a new FormData object to send data to the server
        const formData = new FormData();
        // Append the dataset type ('rgb') to the FormData object
        formData.append('datasetType', 'rgb');

        // Send a POST request to the '/extract_auto' endpoint
        fetch('/geoimaging/extract_auto', {
            method: 'POST',
            body: formData
        })
            .then(response => response.json()) // Parse the response as JSON
            .then(data => {
                // Check if the extraction was successful
                if (data.success) {
                    console.log('RGB dataset extracted successfully');
                    // Perform additional actions after successful extraction, if needed
                } else {
                    console.error('Error extracting RGB dataset:', data.error);
                    // Handle the error case, e.g., display an error message to the user
                }
            })
            .catch(error => {
                console.error('Error communicating with the server:', error);
                // Handle network or other communication errors
            });
    });

    // Open the modal when the "Extract Dataset Manual" button is clicked
    const rgbExtractManualBtn = document.getElementById('rgbExtractManualBtn');
    if (rgbExtractManualBtn) {
        rgbExtractManualBtn.addEventListener('click', function () {
            loadModal();
        });
    } else {
        console.error('Element with ID "rgbExtractManualBtn" not found');
    }

    function loadModal() {
        fetch('/geoimaging/static/rgb_extract_manual.html')
            .then(response => response.text())
            .then(html => {
                const popupContainer = document.getElementById('popupContainer');
                popupContainer.innerHTML = '';
                popupContainer.innerHTML = html;

                const manualExtractionModal = document.getElementById('manualExtractionModal');

                // Add the event listener to the modal
                manualExtractionModal.addEventListener('shown.bs.modal', function () {
                    // Get references to the modal elements
                    const viSelect = document.getElementById('viSelect');
                    const prevPageBtn = document.getElementById('prevPageBtn');
                    const nextPageBtn = document.getElementById('nextPageBtn');
                    const okBtn = document.getElementById('okBtn');
                    const showHistogramBtn = document.getElementById('showHistogramBtn');
                    const showColorCompositeBtn = document.getElementById('showColorCompositeBtn');

                    // Define the showHistogram function
                    // Show histogram for a specific plot
                    function showHistogram(plotIndex) {
                        if (selectedPlotIndex !== null) {
                            const formData = new FormData();
                            formData.append('vi', selectedVI);
                            formData.append('plot_index', plotIndex);
                            formData.append('hist_or_cc', 'true');

                            fetch('/geoimaging/get_rgb_histogram_data', {
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
                    // Show color composite for a specific plot
                    function showColorComposite(plotIndex) {
                        if (selectedPlotIndex !== null) {
                            const formData = new FormData();
                            formData.append('vi', selectedVI);
                            formData.append('plot_index', plotIndex);
                            formData.append('hist_or_cc', 'false');

                            fetch('/geoimaging/get_rgb_histogram_data', {
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

                    // Fetch the initial data (list of VIs and plot data for the first page)
                    function fetchInitialData() {
                        fetch('/geoimaging/rgb_manual_extraction')
                            .then(response => response.json())
                            .then(data => {
                                if (data.error) {
                                    console.error('Error:', data.error);
                                } else {
                                    populateVIDropdown(data.vis);
                                    renderPlotData(data.plotData, 1);
                                }
                            })
                            .catch(error => {
                                console.error('Error:', error);
                            });
                    }

                    // Populate the VI dropdown
                    function populateVIDropdown(vis) {
                        viSelect.innerHTML = '';
                        vis.forEach(vi => {
                            const option = document.createElement('option');
                            option.value = vi;
                            option.text = vi;
                            viSelect.add(option);
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

                    // Handle VI dropdown change
                    viSelect.addEventListener('change', function () {
                        const selectedVI = this.value;
                        fetchPlotData(selectedVI, 1);
                    });

                    // Handle pagination buttons
                    prevPageBtn.addEventListener('click', function () {
                        const currentPage = parseInt(document.getElementById('currentPageDisplay').textContent.split(' ')[1]);
                        if (currentPage > 1) {
                            const selectedVI = document.getElementById('viSelect').value;
                            fetchPlotData(selectedVI, currentPage - 1);
                        }
                    });

                    nextPageBtn.addEventListener('click', function () {
                        const currentPage = parseInt(document.getElementById('currentPageDisplay').textContent.split(' ')[1]);
                        const selectedVI = document.getElementById('viSelect').value;
                        fetchPlotData(selectedVI, currentPage + 1);
                    });

                    // Fetch plot data for the selected VI and page
                    function fetchPlotData(vi, page) {
                        const url = `/geoimaging/get_rgb_plot_data?vi=${vi}&page=${page}`;
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
                        const selectedVI = document.getElementById('viSelect').value;
                        const thresholds = Array.from(document.getElementById('plotDataTableBody').rows).map(row => row.cells[2].querySelector('input[type="range"]').value);

                        const formData = new FormData();
                        formData.append('vi', selectedVI);
                        thresholds.forEach(threshold => {
                            formData.append('thresholds[]', threshold);
                        });

                        fetch('/geoimaging/rgb_manual_extraction_ok', {
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
                });

                $(manualExtractionModal).modal('show');
            })
            .catch(error => {
                console.error('Error loading modal:', error);
            });
    }
});
