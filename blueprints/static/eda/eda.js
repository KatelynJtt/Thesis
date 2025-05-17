/**
 * EDA (Exploratory Data Analysis) Module
 * 
 * This module is wrapped in an Immediately Invoked Function Expression (IIFE)
 * to create a closure. This approach ensures that variables like 'originalDataset'
 * and 'editedDataset' persist across page reloads without being redeclared.
 * It also prevents polluting the global scope while still making the initializeEDA
 * function globally accessible for initialization when the EDA tab is loaded.
 */

(function() {
    let originalDataset = null;
    let editedDataset = null;

    window.initializeEDA = function() {
        // Check script loaded
        console.log('EDA script loaded');
        const filename = getSelectedFile();

        if (!filename) {
            displayMessage('Please select a file from the source column to use EDA functions.', true);
            disableEDAFunctions();
        } else {
            displayMessage('File selected. EDA functions are ready to use.');
            enableEDAFunctions();
            processSelectedFile();
        }
        
        // FILE UPLOAD //////////////////////////////////////////////////////////////////////////////////
        document.querySelectorAll('.menu-item').forEach((item, index) => {
            item.addEventListener('click', function() {
                const targetId = this.getAttribute('data-target');
                document.getElementById(targetId).scrollIntoView({behavior: 'smooth'});
            });
        });
        
        document.querySelectorAll('.save-btn').forEach(btn => {
            btn.addEventListener('click', function() {
                const section = this.closest('.section');
                const sectionId = section.id;
                const displayArea = section.querySelector('.display-area');
                saveFigure(sectionId, displayArea);
                // debugging
            });
        });

        // SHOW DATASET //////////////////////////////////////////////////////////////////////////////
        document.getElementById('show-original-dataset').addEventListener('change', showOriginalDataset)
        document.getElementById('show-edited-dataset').addEventListener('change', showEditedDataset)

        // DATASET ANALYSIS OVERVIEW /////////////////////////////////////////////////////////////////////////
        document.getElementById('analysis-overview-btn').addEventListener('click', function() {
            document.getElementById('show-analysis-overview').checked = true;
            performAnalysisOverview();
        });

        document.getElementById('show-analysis-overview').addEventListener('change', function() {
            document.getElementById('analysis-overview-results').style.display = this.checked ? 'flex' : 'none';
        });

        // Add event listeners to buttons
        document.getElementById('dataset-info-btn').addEventListener('click', showDatasetInfo);
        document.getElementById('statistical-summary-btn').addEventListener('click', showStatisticalSummary);
        document.getElementById('missing-values-btn').addEventListener('click', showMissingValues);
        document.getElementById('outliers-btn').addEventListener('click', showOutliers);
        document.getElementById('vif-btn').addEventListener('click', showVIF);

        // Add event listeners for analyze buttons
        document.getElementById('univariate-analyze-btn').addEventListener('click', performUnivariateAnalysis);
        document.getElementById('bivariate-analyze-btn').addEventListener('click', performBivariateAnalysis);
        document.getElementById('multivariate-analyze-btn').addEventListener('click', performMultivariateAnalysis);
    }

    // Call initializeEDA when the script loads
    initializeEDA();
    
    function handleResponse(response) {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const contentType = response.headers.get("content-type");
        if (contentType && contentType.indexOf("application/json") !== -1) {
            return response.json();
        } else {
            throw new Error("Oops, we haven't got JSON!");
        }
    }    

    function saveFigure(sectionId, displayArea) {
        // Implement saving logic here
        console.log(`Saving figure from ${sectionId}`);
    }

    function getSelectedFile() {
        const selectedFileBtn = document.querySelector('.file-btn.selected');
        return selectedFileBtn ? selectedFileBtn.dataset.filename : null;
    }

    function displayMessage(message, isError = false) {
        const messageElement = document.getElementById('eda-message') || document.createElement('div');
        messageElement.id = 'eda-message';
        messageElement.textContent = message;
        messageElement.style.color = isError ? 'red' : 'green';
        messageElement.style.padding = '10px';
        messageElement.style.marginBottom = '20px';
        const edaContent = document.getElementById('eda-content');
        edaContent.insertBefore(messageElement, edaContent.firstChild);
    }

    function disableEDAFunctions() {
        document.querySelectorAll('#eda-content button, #eda-content select').forEach(el => el.disabled = true);
    }

    function enableEDAFunctions() {
        document.querySelectorAll('#eda-content button, #eda-content select, #show-original-dataset').forEach(el => el.disabled = false);
    }

    // FILE PROCESSING //////////////////////////////////////////////////////////////////////////////////
    function processSelectedFile() {
        const filename = getSelectedFile();
        if (filename) {
            fetch(`/eda/process_file?file=${filename}`)
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        displayTable(data.data, {}, 'original-dataset-display');
                        document.getElementById('show-original-dataset').checked = true;
                        enableAnalysisOverview();
                        populateFeatureDropdowns(Object.keys(data.data[0]));
                        showRecommendations();
                    } else {
                        displayMessage('Error: ' + data.error, true);
                    }
                });
        }
    }

    function showOriginalDataset() {
        const checkbox = document.getElementById('show-original-dataset');
        const displayArea = document.getElementById('original-dataset-display');
    
        if (checkbox.checked) {
            if (originalDataset) {
                displayArea.style.display = 'block';
            } else {
                fetch('/eda/show_dataset')
                    .then(response => response.json())
                    .then(data => {
                        if (data.error) {
                            displayMessage(data.error, true);
                        } else {
                            originalDataset = data;
                            displayTable(data, {}, 'original-dataset-display');
                            displayArea.style.display = 'block';
                        }
                    })
                    .catch(error => {
                        console.error('Error:', error);
                        displayMessage('An error occurred while fetching the dataset.', true);
                    });
            }
        } else {
            displayArea.style.display = 'none';
        }
    }    

    function showEditedDataset() {
        const checkbox = document.getElementById('show-edited-dataset');
        const displayArea = document.getElementById('edited-dataset-display');
    
        if (checkbox.checked) {
            if (editedDataset) {
                displayArea.style.display = 'block';
            } else {
                fetch('/eda/show_edited_dataset')
                    .then(response => response.json())
                    .then(data => {
                        if (data.error) {
                            displayMessage(data.error, true);
                        } else {
                            editedDataset = data;
                            displayTable(data, {}, 'edited-dataset-display');
                            displayArea.style.display = 'block';
                        }
                    })
                    .catch(error => {
                        console.error('Error:', error);
                        displayMessage('An error occurred while fetching the edited dataset.', true);
                    });
            }
        } else {
            displayArea.style.display = 'none';
        }
    }
    
    function showDatasetInfo() {
        const filename = getSelectedFile();
        fetch(`/eda/dataset_info?file=${filename}`)
        .then(handleResponse)
        .then(data => {
            if (data.error) {
                displayMessage(data.error, true);
            } else {
                const displayArea = document.getElementById('analysis-display-area');
                displayArea.innerHTML = `<pre>${data.dataset_info}</pre>`;
            }
        })
        .catch(error => {
            console.error('Error:', error);
            displayMessage('An error occurred while fetching dataset info.', true);
        });
    }
    
    function showStatisticalSummary() {
        const filename = getSelectedFile();
        fetch(`/eda/statistical_summary?file=${filename}`)
        .then(handleResponse)
        .then(data => {
            const table = createTable(data);
            const displayArea = document.getElementById('analysis-display-area');
            displayArea.innerHTML = '';
            displayArea.appendChild(table);
        })
        .catch(error => {
            console.error('Error:', error);
            displayMessage('An error occurred while fetching statistical summary.', true);
        });
    }
    
    function showMissingValues() {
        const filename = getSelectedFile();
        fetch(`/eda/missing_values?file=${filename}`)
        .then(handleResponse)
        .then(data => {
            const displayArea = document.getElementById('analysis-display-area');
            displayArea.innerHTML = '';
            if (data.error) {
                displayMessage(data.error, true);
            } else if (!data || data.length === 0) {
                displayArea.textContent = 'No missing values found';
            } else {
                displayArea.appendChild(createTable(data));
            }
        })
        .catch(error => {
            console.error('Error:', error);
            displayMessage('An error occurred while fetching missing values data.', true);
        });
    }
    
    function showOutliers() {
        const filename = getSelectedFile();
        fetch(`/eda/outliers?file=${filename}`)
        .then(handleResponse)
        .then(data => {
            const displayArea = document.getElementById('analysis-display-area');
            displayArea.innerHTML = '';
            if (data.error) {
                displayMessage(data.error, true);
            } else if (!data || Object.keys(data).length === 0) {
                displayArea.textContent = 'No outliers found';
            } else {
                const tableData = Object.entries(data).flatMap(([feature, outliers]) =>
                    Object.entries(outliers).map(([index, value]) => ({Feature: feature, Index: index, Value: value}))
                );
                displayArea.appendChild(createTable(tableData));
            }
        })
        .catch(error => {
            console.error('Error:', error);
            displayMessage('An error occurred while fetching outliers data.', true);
        });
    }
    
    function showVIF() {
        const filename = getSelectedFile();
        fetch(`/eda/vif?file=${filename}`)
        .then(handleResponse)
        .then(data => {
            const displayArea = document.getElementById('analysis-display-area');
            displayArea.innerHTML = '';
            if (data.error) {
                displayMessage(data.error, true);
            } else {
                const table = createTable(data.vif_data);
                const img = document.createElement('img');
                img.src = 'data:image/png;base64,' + data.plot_url;
                img.className = 'responsive-chart';
                displayArea.appendChild(table);
                displayArea.appendChild(img);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            displayMessage('An error occurred while fetching VIF data.', true);
        });
    }
    
    function showRecommendations() {
        const filename = getSelectedFile();
        fetch(`/eda/recommendations?file=${filename}`)
        .then(handleResponse)
        .then(data => {
            const recommendationsList = document.getElementById('recommendations-list');
            recommendationsList.innerHTML = '';
            data.recommendations.forEach(recommendation => {
                const li = document.createElement('li');
                
                const button = document.createElement('button');
                button.textContent = 'Apply';
                button.onclick = () => applyRecommendation(filename, recommendation.type, recommendation.features);
                
                li.appendChild(button);
                li.appendChild(document.createTextNode(' ' + recommendation.text));
                
                recommendationsList.appendChild(li);
            });
        })
        .catch(error => {
            console.error('Error:', error);
            displayMessage('An error occurred while fetching recommendations.', true);
        });
    }
    
    function applyRecommendation(filename, type, features) {
        let action;
        switch(type) {
            case 'missing_values':
                action = prompt('Choose strategy for missing values: mean, median, mode, or drop');
                break;
            case 'outliers':
                action = prompt('Choose method for outliers: iqr or zscore');
                break;
            case 'scaling':
                action = prompt('Choose scaling method: standard or minmax');
                break;
            case 'high_vif':
                action = prompt('Choose action for high VIF: remove or transform');
                break;
            case 'categorical':
                action = prompt('Choose encoding method: onehot or label');
                break;
        }
    
        if (action) {
            fetch('/eda/apply_recommendation', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({file: filename, type, features, action}),
            })
            .then(handleResponse)
            .then(data => {
                if (data.success) {
                    displayMessage('Recommendation applied successfully');
                    datasetEdited = true;
                    document.getElementById('show-edited-dataset').disabled = false;
                    showEditedDataset(data.changes);
                } else {
                    displayMessage('Error: ' + data.error, true);
                }
            })
            .catch(error => {
                console.error('Error:', error);
                displayMessage('An error occurred while applying the recommendation.', true);
            });
        }
    }
    
    function saveResults() {
        const results = document.getElementById('analysis-display-area').innerHTML;
        const filename = prompt("Enter a filename to save the results:", "analysis_results");
        const selectedFile = getSelectedFile();
        
        if (filename) {
            fetch('/eda/save_results', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({results: results, filename: filename, file: selectedFile}),
            })
            .then(handleResponse)
            .then(data => {
                displayMessage(data.message);
            })
            .catch(error => {
                console.error('Error:', error);
                displayMessage('An error occurred while saving the results.', true);
            });
        }
    }
    
    // DATASET ANALYSIS OVERVIEW /////////////////////////////////////////////////////////////////////////
    function enableAnalysisOverview() {
        const showAnalysisOverview = document.getElementById('show-analysis-overview');
        if (showAnalysisOverview) {
            showAnalysisOverview.disabled = false;
        }
        document.getElementById('analysis-overview-btn').disabled = false;
        document.getElementById('show-original-dataset').disabled = false;
    }

    function performAnalysisOverview() {
        const filename = getSelectedFile();
    
        if (!filename) {
            displayMessage('Please select a file first.', true);
            return;
        }
    
        Promise.all([
            fetch(`/eda/pie_charts?file=${filename}`).then(handleResponse),
            fetch(`/eda/vif?file=${filename}`).then(handleResponse)
        ])
        .then(([pieChartsData, vifData]) => {
            console.log('Pie charts data:', pieChartsData);
            console.log('VIF data:', vifData);
    
            if (pieChartsData.missing_plot_url) {
                document.getElementById('missing-values-chart').src = 'data:image/png;base64,' + pieChartsData.missing_plot_url;
            } else {
                console.error('Missing values plot URL is undefined');
            }
    
            if (pieChartsData.outliers_plot_url) {
                document.getElementById('outliers-chart').src = 'data:image/png;base64,' + pieChartsData.outliers_plot_url;
            } else {
                console.error('Outliers plot URL is undefined');
            }
    
            if (vifData.plot_url) {
                document.getElementById('vif-chart').src = 'data:image/png;base64,' + vifData.plot_url;
            } else {
                console.error('VIF plot URL is undefined');
            }
    
            document.getElementById('analysis-overview-results').style.display = 'flex';
        })
        .catch(error => {
            console.error('Error:', error);
            displayMessage('An error occurred while fetching the analysis overview.', true);
        });
    }           

    // Dropdown menu for feature selection for analysis
    function populateFeatureDropdowns(features) {
        const univariateSelect = document.getElementById('univariate-feature-select');
        const bivariateSelect1 = document.getElementById('bivariate-feature-select-1');
        const bivariateSelect2 = document.getElementById('bivariate-feature-select-2');
        const multivariateSelect = document.getElementById('multivariate-feature-select');

        features.forEach(feature => {
            univariateSelect.add(new Option(feature, feature));
            bivariateSelect1.add(new Option(feature, feature));
            bivariateSelect2.add(new Option(feature, feature));
            multivariateSelect.add(new Option(feature, feature));
        });
    }

    // Univariate analysis function
    function performUnivariateAnalysis() {
        const featureSelect = document.getElementById('univariate-feature-select');
        const filename = getSelectedFile();
    
        if (!featureSelect || !featureSelect.value) {
            displayMessage('Please select a feature for univariate analysis.', true);
            return;
        }
        
        fetch(`/eda/univariate?file=${filename}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({feature: featureSelect.value}),
        })
        .then(handleResponse)
        .then(data => {
            const displayArea = document.querySelector('#univariate .display-area');
            displayArea.innerHTML = `<img src="data:image/png;base64,${data.plot_url}" alt="Univariate Analysis Plot" class="responsive-chart">`;
        })
        .catch(error => {
            console.error('Error:', error);
            displayMessage('An error occurred during univariate analysis.', true);
        });
    }
    
    // Bivariate analysis function
    function performBivariateAnalysis() {
        const feature1Select = document.getElementById('bivariate-feature-select-1');
        const feature2Select = document.getElementById('bivariate-feature-select-2');
        const filename = getSelectedFile();
    
        if (!feature1Select || !feature1Select.value || !feature2Select || !feature2Select.value) {
            displayMessage('Please select two features for bivariate analysis.', true);
            return;
        }
    
        fetch(`/eda/bivariate?file=${filename}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({feature_1: feature1Select.value, feature_2: feature2Select.value}),
        })
        .then(handleResponse)
        .then(data => {
            const displayArea = document.querySelector('#bivariate .display-area');
            displayArea.innerHTML = `<img src="data:image/png;base64,${data.plot_url}" alt="Bivariate Analysis Plot" class="responsive-chart">`;
        })
        .catch(error => {
            console.error('Error:', error);
            displayMessage('An error occurred during bivariate analysis.', true);
        });
    }
    
    // Multivariate analysis function
    function performMultivariateAnalysis() {
        const featureSelect = document.getElementById('multivariate-feature-select');
        const selectedFeatures = Array.from(featureSelect.selectedOptions).map(option => option.value);
        const filename = getSelectedFile();
    
        if (selectedFeatures.length < 2) {
            displayMessage('Please select at least two features for multivariate analysis.', true);
            return;
        }
    
        fetch(`/eda/multivariate?file=${filename}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({features: selectedFeatures})
        })
        .then(handleResponse)
        .then(data => {
            displayMultivariateResults(data);
        })
        .catch(error => {
            console.error('Error:', error);
            displayMessage('An error occurred during multivariate analysis.', true);
        });
    }
            
    function displayMultivariateResults(data) {
        const displayArea = document.querySelector('#multivariate .display-area');
        displayArea.innerHTML = `
            <div style="display: flex; flex-direction: column; align-items: center;">
                <div style="width: 100%; margin-bottom: 20px;">
                    <h3>Variance Inflation Factor (VIF)</h3>
                    <img src="data:image/png;base64,${data.vif_plot_url}" style="width: 100%; max-width: 800px;">
                </div>
                <div style="width: 100%;">
                    <h3>Correlation Matrix</h3>
                    <img src="data:image/png;base64,${data.corr_plot_url}" style="width: 100%; max-width: 800px;">
                </div>
            </div>
        `;
    }    

    function saveFigure(sectionId, displayArea) {
        // Implement saving logic here
        const content = displayArea.innerHTML;
        const filename = `${sectionId}_analysis.html`;
        
        const blob = new Blob([content], {type: 'text/html'});
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = filename;
        link.click();
    }

    // DISPLAY TABLE
    function displayTable(data, changes = {}, displayAreaId) {
        const table = document.createElement('table');
        const thead = document.createElement('thead');
        const tbody = document.createElement('tbody');

        // Create table header
        const headerRow = document.createElement('tr');
        Object.keys(data[0]).forEach(key => {
            const th = document.createElement('th');
            th.textContent = key;
            headerRow.appendChild(th);
        });
        thead.appendChild(headerRow);
        table.appendChild(thead);

        // Create table body
        data.forEach((row, rowIndex) => {
            const tr = document.createElement('tr');
            Object.entries(row).forEach(([key, value]) => {
                const td = document.createElement('td');
                td.textContent = value;
                if (changes[key] && changes[key].includes(rowIndex)) {
                    td.style.backgroundColor = 'yellow';
                }
                tr.appendChild(td);
            });
            tbody.appendChild(tr);
        });
        table.appendChild(tbody);

        document.getElementById(displayAreaId).innerHTML = '';
        document.getElementById(displayAreaId).appendChild(table);
    }

    function createTable(data) {
        if (!data || data.length === 0) {
            return document.createTextNode('No data available');
        }
        
        const table = document.createElement('table');
        table.className = 'eda-table';
        const thead = document.createElement('thead');
        const tbody = document.createElement('tbody');

        const headerRow = document.createElement('tr');
        Object.keys(data[0]).forEach(key => {
            const th = document.createElement('th');
            th.textContent = key;
            headerRow.appendChild(th);
        });
        thead.appendChild(headerRow);
        table.appendChild(thead);

        data.forEach((row, index) => {
            const tr = document.createElement('tr');
            tr.className = index % 2 === 0 ? 'even-row' : 'odd-row';
            Object.values(row).forEach(value => {
                const td = document.createElement('td');
                td.textContent = value;
                tr.appendChild(td);
            });
            tbody.appendChild(tr);
        });
        table.appendChild(tbody);

        return table;
    }
})();