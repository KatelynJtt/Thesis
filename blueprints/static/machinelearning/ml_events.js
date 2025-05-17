/**
 * Machine Learning Module
 * 
 * This script manages the machine learning functionality of the application.
 * It handles feature selection, model parameter fetching, and UI updates
 * related to machine learning tasks. The script is encapsulated in an IIFE
 * to avoid polluting the global scope, with only the initialization function
 * exposed globally.
 */

(function() {
    let modelParams = {};
    let maxModels = 1;
    let searchMethod = 'default';
    let selectedModels = [];
    let ensembleMethod = 'none';

    // Initialize the machine learning module
    window.initializeMachineLearning = function() {
        console.log('Machine Learning script loaded');

        // Check if DOM is ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', setupEventListeners);
        } else {
            setupEventListeners();
        }  
    };

    function setupEventListeners() {
        const filename = getSelectedFile();
        console.log('Setting up event listeners with filename:', filename);

        // Log DOM elements we're trying to access
        console.log('Form element:', document.querySelector('form'));
        console.log('Test size input:', document.getElementById('test_size'));
        console.log('Model type toggle:', document.getElementById('modelTypeToggle'));

        // Toggle functionality - always active
        const modelTypeToggle = document.getElementById('modelTypeToggle');
        if (modelTypeToggle) {
            modelTypeToggle.addEventListener('change', handleModelTypeToggle);
        }
    
        if (!filename) {
            displayMessage('Please select a file to use Machine Learning functions.', true);
            disableMLFunctions();
        } else {
            displayMessage('File selected. Machine Learning functions are ready to use.');
            enableMLFunctions();
            
            // Add event listeners
            const form = document.querySelector('form');
            if (form) {
                form.addEventListener('submit', handleFormSubmit);
            }
            if (document.querySelector('.test-size-container')) {
                createNotches();
            }
            const testSizeInput = document.getElementById('test_size');
            if (testSizeInput) {
                testSizeInput.addEventListener('input', updateTestSizeValue);
            }
            const modelTypeToggle = document.getElementById('modelTypeToggle');
            if (modelTypeToggle) {
                modelTypeToggle.addEventListener('change', handleModelTypeToggle);
            }
            const ensembleRadios = document.querySelectorAll('input[name="ensemble_method"]');
            if (ensembleRadios) {
                ensembleRadios.forEach(input => {
                    input.addEventListener('change', handleEnsembleMethodChange);
                });
            }
            const methodCountInput = document.getElementById('methodCount');
            if (methodCountInput) {
                methodCountInput.addEventListener('change', function() {
                    maxModels = parseInt(this.value);
                    updateModelCount(maxModels);
                    
                    // Remove excess selected models if the new max is lower
                    while (selectedModels.length > maxModels) {
                        const modelToRemove = selectedModels.pop();
                        const button = document.querySelector(`button:not(.runButton)[data-model="${modelToRemove}"]`);
                        if (button) {
                            button.classList.remove('selected');
                            button.style.backgroundColor = '';
                        }
                        document.querySelector(`#modelParametersContainer .model-params[data-model="${modelToRemove}"]`)?.remove();
                    }
                });
            }
            const regressionBtnGroup = document.getElementById('regressionBtnGroup');
            if (regressionBtnGroup) {
                regressionBtnGroup.addEventListener('click', handleModelSelection);
            }
            const classificationBtnGroup = document.getElementById('classificationBtnGroup');
            if (classificationBtnGroup) {
                classificationBtnGroup.addEventListener('click', handleModelSelection);
            }
            const runButtonRegression = document.querySelector('#regressionBtnGroup #runButton');
            if (runButtonRegression) {
                runButtonRegression.addEventListener('click', handleRunButtonClick);
            }
            const runButtonClassification = document.querySelector('#classificationBtnGroup #runButton');
            if (runButtonClassification) {
                runButtonClassification.addEventListener('click', handleRunButtonClick);
            }
            const runAllButton = document.getElementById('runAllModels');
            if (runAllButton) {
                runAllButton.addEventListener('click', handleRunAllModels);
            }
            const saveDisplayedResult = document.getElementById('saveDisplayedResult');
            if (saveDisplayedResult) {
                saveDisplayedResult.addEventListener('click', handleSaveDisplayedResult);
            }
            const viewOptimalParams = document.getElementById('viewOptimalParams');
            if (viewOptimalParams) {
                viewOptimalParams.addEventListener('click', handleViewOptimalParams);
            }
            // Add custom input change listener
            const paramSelects = document.querySelectorAll('.param-select');
            if (paramSelects) {
                paramSelects.forEach(select => {
                    select.addEventListener('change', handleCustomInputChange);
                });
            }
            const explainShap = document.getElementById('explainShap');
            if (explainShap) {
                explainShap.addEventListener('click', handleExplainShap);
            }
            const explainLime = document.getElementById('explainLime');
            if (explainLime) {
                explainLime.addEventListener('click', handleExplainLime);
            }
            const helpButton = document.getElementById('helpButton');
            if (helpButton) {
                helpButton.addEventListener('click', handleHelpButton);
            }
            const closeButton = document.getElementById('closeButton');
            if (closeButton) {
                closeButton.addEventListener('click', handleCloseButton);
            }
            document.getElementById('saveTrainedModel').addEventListener('click', handleSaveTrainedModel);
            document.getElementById('saveDisplayedResult').addEventListener('click', handleSaveDisplayedResult);


            // Update file info and initialize tabs
            updateSelectedFile(filename);
            initializeTabs();
        }
    }

    // Call initializeMachineLearning when the script loads
    initializeMachineLearning();
    

    function handleCustomInputChange(event) {
        if (event.target.classList.contains('param-select')) {
            const customInput = event.target.nextElementSibling;
            if (customInput && customInput.classList.contains('custom-input')) {
                customInput.style.display = event.target.value === 'other' ? 'inline' : 'none';
            }
        }
    }

    // Update the displayed test size value
    function updateTestSizeValue(event) {
        const value = event.target.value;
        const valueDisplay = document.getElementById('test_size_value');
        const slider = event.target;
        const percent = (value - slider.min) / (slider.max - slider.min);
        const sliderWidth = slider.offsetWidth;
        const valueWidth = valueDisplay.offsetWidth;
        const leftOffset = percent * (sliderWidth - valueWidth) + valueWidth / 2;
        
        valueDisplay.style.left = `${leftOffset}px`;
        valueDisplay.textContent = value;
    }
    
    function createNotches() {
        const track = document.createElement('div');
        track.className = 'test-size-track';
        
        for (let i = 0; i <= 10; i++) {
            const notch = document.createElement('div');
            notch.className = 'notch';
            notch.style.left = `${i * 10}%`;
            track.appendChild(notch);
        }
        
        const container = document.querySelector('.test-size-container');
        container.insertBefore(track, document.getElementById('test_size'));
    }    

    // Initialize tab display
    function initializeTabs() {
        document.querySelectorAll('.tab-pane').forEach(pane => pane.style.display = 'none');
        document.querySelector('.tab-pane.active').style.display = 'block';
    }

    // Handle model selection and parameter fetching
    function handleModelSelection(event) {
        if (event.target.matches('button:not(.runButton)')) {
            const model = event.target.textContent;
            const runButton = event.target.parentElement.querySelector('.runButton');
            
            if (event.target.classList.contains('selected')) {
                event.target.classList.remove('selected');
                selectedModels = selectedModels.filter(m => m !== model);
                document.querySelector(`#modelParametersContainer .model-params[data-model="${model}"]`)?.remove();
            } else if (selectedModels.length < maxModels) {
                event.target.classList.add('selected');
                selectedModels.push(model);
                fetchModelParams(model)
                    .then(data => {
                        const paramHTML = createParamHTML(model, data);
                        document.getElementById('modelParametersContainer').insertAdjacentHTML('beforeend', paramHTML);
                    });
            } else {
                alert(`You've reached the maximum number of models (${maxModels}).`);
            }
    
            // Update run button color
            if (selectedModels.length === maxModels) {
                runButton.classList.add('ready');
            } else {
                runButton.classList.remove('ready');
            }
            
            console.log("Selected models:", selectedModels);
        }
    }
    
    function handleModelTypeToggle() {
        const regressionContent = document.getElementById('regression');
        const classificationContent = document.getElementById('classification');
        const regressionLabel = document.querySelector('.toggle-label:first-child');
        const classificationLabel = document.querySelector('.toggle-label:last-child');
    
        console.log('Regression content:', regressionContent);
        console.log('Classification content:', classificationContent);
    
        if (this.checked) {
            console.log('Showing Classification');
            regressionContent.style.display = 'none';
            classificationContent.style.display = 'block';
            regressionLabel.style.opacity = '0.5';
            classificationLabel.style.opacity = '1';
        } else {
            console.log('Showing Regression');
            regressionContent.style.display = 'block';
            classificationContent.style.display = 'none';
            regressionLabel.style.opacity = '1';
            classificationLabel.style.opacity = '0.5';
        }
    }

    // Get SELECTED FILE from the UI ///////////////////////////////////////////////////////////////////////
    function updateSelectedFile() {
        const filename = getSelectedFile();
        const uploadPrompt = document.getElementById('upload-prompt');
        const fileInfo = document.getElementById('file-info');
        const selectedFilename = document.getElementById('selected-filename');
        
        if (filename) {
            uploadPrompt.style.display = 'none';
            fileInfo.style.display = 'block';
            selectedFilename.textContent = filename;
            fetchFeatures(filename);
        } else {
            uploadPrompt.style.display = 'block';
            fileInfo.style.display = 'none';
            document.getElementById('features').innerHTML = '';
            document.getElementById('label').innerHTML = '';
        }
    }

    function getSelectedFile() {
        const selectedFileBtn = document.querySelector('.file-btn.selected');
        return selectedFileBtn ? selectedFileBtn.dataset.filename : null;
    }
    
    function displayMessage(message, isError = false) {
        const messageElement = document.getElementById('message-div');
        if (messageElement) {
            messageElement.textContent = message;
            messageElement.style.color = isError ? 'red' : 'green';
        }
    }
    
    function disableMLFunctions() {
        document.querySelectorAll('#ml-content button, #ml-content select').forEach(el => {
            if (el) el.disabled = true;
        });
    }
    
    function enableMLFunctions() {
        document.querySelectorAll('#ml-content button, #ml-content select').forEach(el => {
            if (el) el.disabled = false;
        });
    }
    
    // Fetch features for the selected file from the server
    function fetchFeatures(filename) {
        console.log('Fetching features for:', filename);
        
        fetch(`/machinelearning/get_columns?file=${filename}`)
            .then(response => response.json())
            .then(data => {
                console.log('Received columns:', data.columns);
                if (data.columns && data.columns.length > 0) {
                    populateDropdowns(data.columns);
                }
            })
            .catch(error => console.error('Error:', error));
    }

    // Populate feature and label dropdowns with the fetched columns
    function populateDropdowns(columns) {
        const featureSelect = document.getElementById('features');
        const labelSelect = document.getElementById('label');
        
        // Clear existing options
        featureSelect.innerHTML = '';
        labelSelect.innerHTML = '';
        
        // Add new options
        columns.forEach(column => {
            const featureOption = new Option(column, column);
            const labelOption = new Option(column, column);
            
            featureSelect.add(featureOption);
            labelSelect.add(labelOption);
        });
    }

    // Handle form submission for feature and label selection
    function handleFormSubmit(e) {
        e.preventDefault();
        const selectedFeatures = Array.from(document.getElementById('features').selectedOptions).map(option => option.value);
        const selectedLabel = document.getElementById('label').value;

        fetch('/machinelearning/select_features', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                filename: getSelectedFile(),
                features: selectedFeatures,
                label: selectedLabel
            })
        })
        .then(response => response.json())
        .then(result => {
            displayMessage(result.message);
        })
        .catch(error => {
            console.error('Error:', error);
            displayMessage('An error occurred while selecting features and label.');
        });
    }

    // Fetch parameters for a specific model from the server
    function fetchModelParams(model) {
        const url = `/machinelearning/get_model_params/${model}`;
        console.log('Fetching from URL:', url);
        return fetch(url)
            .then(response => {
                console.log('Response status:', response.status);
                return response.json();
            })
            .catch(error => {
                console.error('Fetch error:', error);
            });
    }    

    // Fetch layout for a specific model considering the current search method
    function fetchModelLayout(model) {
        return fetch(`/machinelearning/get_model_layout/${model}/${searchMethod}`)
            .then(response => response.json())
            .then(data => {
                console.log('Fetched layout data:', data);
                return data;
            });
    }

    // Update the displayed count of models to be selected
    function updateModelCount(count) {
        maxModels = count;
        const methodCountInput = document.getElementById('methodCount');
        if (methodCountInput) {
            methodCountInput.value = count;
        }
        console.log("Updated max models to:", maxModels);
    }    

    // Reset the maximum number of models and clear model parameters
    function resetMaxModels() {
        maxModels = 1;
        modelParams = {};
        updateModelCount(maxModels);
        console.log("Resetting max model number to:", maxModels);
        console.log("Resetting model parameters");
    }

    /// Create HTML for model parameters
    function createParamHTML(model, params) {
        let html = `<div class="model-params" data-model="${model}">`;
        html += `<h3>${model}</h3>`;
        for (const [key, value] of Object.entries(params)) {
            html += `
                <div class="param-row">
                    <label>${key}: </label>
                    <select name="${model}-${key}" class="param-select">
                        ${createOptions(value, key)}
                    </select>
                    ${isInteger(value) ? `<input type="number" class="custom-input" style="display:none;">` : ''}
                </div>`;
        }
        html += '</div>';
        return html;
    }
    
    // Create options for the select dropdowns
    function createOptions(value, key) {
        let options = '';
        if (Array.isArray(value)) {
            value.forEach(v => options += `<option value="${v}">${v}</option>`);
            if (value.every(v => Number.isInteger(Number(v)))) {
                options += '<option value="other">Other</option>';
            }
        } else if (typeof value === 'object') {
            Object.keys(value).forEach(v => options += `<option value="${v}">${v}</option>`);
        }
        return options;
    }
    
    // Check if the value is an integer
    function isInteger(value) {
        return Array.isArray(value) && value.every(v => Number.isInteger(v));
    }
    
    // Manages the behavior when the "Run" button is clicked
    function handleRunButtonClick(event) {
        const isRegression = event.target.closest('#regressionBtnGroup') !== null;
        const route = isRegression ? '/machinelearning/get_regression_modal/' : '/machinelearning/get_classification_modal/';
        const bottomHalf = document.querySelector('.bottom-half');
        
        // Show bottom half
        bottomHalf.style.display = 'block';
        
        // Update heading
        const modelHeading = bottomHalf.querySelector('h2');
        modelHeading.textContent = isRegression ? 'Regressor' : 'Classifier';
        modelHeading.id = isRegression ? 'regressor' : 'classifier';
    
        // Fetch configuration data
        fetch(route)
            .then(response => response.json())
            .then(data => {
                // Apply configuration data
                setupModelParameters(data.parameters);
                setupMetrics(data.metrics);
                console.log("Current Global Values:", JSON.stringify({ selectedModels, isRegression, ensembleMethod }));
                window.MLAppTempStorage = { selectedModels, isRegression, ensembleMethod };
                showModelModal(selectedModels, isRegression, ensembleMethod);
                setupModalEventListeners();
            });
    }    

    // Manages the behavior when the "Run All Models" button is clicked
    function handleRunAllModels() {
        const isRegression = document.getElementById('regression').style.display === 'block';
        const buttonGroup = isRegression ? '#regressionBtnGroup' : '#classificationBtnGroup';
        const allButtons = document.querySelectorAll(`${buttonGroup} button:not(.runButton)`);
        
        maxModels = allButtons.length;
        updateModelCount(maxModels);
        
        selectedModels = [];
        document.querySelectorAll('.button-group button').forEach(btn => btn.classList.remove('selected'));
        document.getElementById('modelParametersContainer').innerHTML = '';
    
        allButtons.forEach(button => {
            button.classList.add('selected');
            const model = button.textContent;
            selectedModels.push(model);
            fetchModelParams(model)
                .then(data => {
                    modelParams[model] = data;
                    const paramHTML = createParamHTML(model, data);
                    document.getElementById('modelParametersContainer').insertAdjacentHTML('beforeend', paramHTML);
                });
        });
    }    

    // Manages the behavior when the "Ensemble Method" dropdown is changed
    function handleEnsembleMethodChange(event) {
        ensembleMethod = event.target.value;
        const methodCountInput = document.getElementById('methodCount');
        const runAllButton = document.getElementById('runAllModels');
        
        if (ensembleMethod === 'none') {
            methodCountInput.value = 1;
            methodCountInput.disabled = true;
            runAllButton.disabled = true;
        } else {
            const activeTab = document.querySelector('#regression').style.display === 'block' ? 'regression' : 'classification';
            const buttonCount = document.querySelectorAll(`#${activeTab}BtnGroup button:not(.runButton)`).length;
            
            methodCountInput.disabled = false;
            methodCountInput.min = 2;
            methodCountInput.max = buttonCount;
            methodCountInput.value = 2;
            runAllButton.disabled = false;
        }
        
        updateModelCount(parseInt(methodCountInput.value));
    }        
    
    /////////////////////////////////////////////////////////////////////////////////////////////////////
    // MODAL WINDOWS ////////////////////////////////////////////////////////////////////////////////////
    function setupModalEventListeners() {
        const bottomHalf = document.querySelector('.bottom-half');
        
        // Search method radio buttons
        const searchMethodInputs = bottomHalf.querySelectorAll('input[name="search_method"]');
        if (searchMethodInputs.length > 0) {
            searchMethodInputs.forEach(input => {
                input.addEventListener('change', handleSearchMethodChange);
            });
        }
    
        // Parameter selection changes
        const modalParams = bottomHalf.querySelector('.modal-params');
        if (modalParams) {
            modalParams.addEventListener('change', handleParamSelectChange);
        }
        
        // Train and predict button
        const trainPredict = bottomHalf.querySelector('#trainPredict');
        if (trainPredict) {
            trainPredict.addEventListener('click', handleTrainPredictClick);
        }
        
        // Additional buttons with null checks
        ['saveDisplayedResult', 'viewOptimalParams', 'explainShap', 'explainLime', 'helpButton'].forEach(id => {
            const element = bottomHalf.querySelector(`#${id}`);
            if (element) {
                element.addEventListener('click', eval(`handle${id.charAt(0).toUpperCase() + id.slice(1)}`));
            }
        });
    }        
    
    function handleSearchMethodChange(event) {
        searchMethod = event.target.value;
        updateModelLayouts();
    }

    function setupModelParameters(parameters) {
        const modelParamsContainer = document.querySelector('.bottom-half .model-params');
        modelParamsContainer.innerHTML = '';
    
        // Log the received parameters for debugging
        console.log('Received parameters:', parameters);
    
        // Get parameters directly from the model layout
        selectedModels.forEach(model => {
            fetchModelLayout(model)
                .then(layout => {
                    const paramHTML = createModalParamsHTML(model, layout);
                    modelParamsContainer.insertAdjacentHTML('beforeend', paramHTML);
                });
        });
    }       
    
    function handleParamSelectChange(event) {
        if (event.target.classList.contains('param-select')) {
            const customInput = event.target.nextElementSibling;
            if (customInput && customInput.classList.contains('custom-input')) {
                customInput.style.display = event.target.value === 'other' ? 'inline' : 'none';
            }
        }
    }
    
    function setupMetrics(metrics) {
        const trainTable = document.querySelector('.bottom-half #trainScoreTable tbody');
        const testTable = document.querySelector('.bottom-half #testScoreTable tbody');
        
        // Clear existing rows
        trainTable.innerHTML = '';
        testTable.innerHTML = '';
        
        // Add metric rows to both tables
        metrics.forEach(metric => {
            const trainRow = `<tr><td>${metric}</td><td></td></tr>`;
            const testRow = `<tr><td>${metric}</td><td></td></tr>`;
            
            trainTable.insertAdjacentHTML('beforeend', trainRow);
            testTable.insertAdjacentHTML('beforeend', testRow);
        });
    }    
    
    function showModelModal(models, regression, ensemble) {
        console.log("Start of showModelModal. Received ensemble:", ensemble);
        
        selectedModels = models;
        isRegression = regression;
        ensembleMethod = ensemble;
        searchMethod = document.querySelector('input[name="search_method"]:checked')?.value || 'none';
        
        console.log("After setting values in showModelModal. Ensemble method:", ensembleMethod);
        console.log("Global Values after show:", { selectedModels, isRegression, ensembleMethod, searchMethod });
    
        updateModelLayouts();
        
        console.log("End of showModelModal. Final ensemble method:", ensembleMethod);
    }    
    
    function updateModelLayouts() {
        Promise.all(selectedModels.map(model => fetchModelLayout(model)))
            .then(layouts => {
                const modelData = selectedModels.map((model, index) => ({
                    model: model,
                    layout: layouts[index]
                }));
                updateModalContent(modelData);
            })
            .catch(error => console.error("Error fetching model layouts:", error));
    }
    
    function updateModalContent(modelData) {
        const bottomHalf = document.querySelector('.bottom-half');
        
        // Set model title and model name
        const modelTitle = ensembleMethod !== 'none' ? ensembleMethod : modelData[0].model;
        const titleElement = bottomHalf.querySelector('.modal-title');
        if (titleElement) {
            titleElement.textContent = modelTitle + (isRegression ? ' Regression' : ' Classification');
        }
        
        const modelNameElement = bottomHalf.querySelector(`#${isRegression ? 'regressor' : 'classifier'}`);
        if (modelNameElement) {
            modelNameElement.textContent = modelTitle;
        }
    
        // Update model-params block
        const modelParamsContainer = bottomHalf.querySelector('.model-params');
        if (modelParamsContainer) {
            modelParamsContainer.innerHTML = '';
    
            if (modelData && modelData.length > 0 && modelData[0].layout) {
                if (ensembleMethod !== 'none') {
                    bottomHalf.querySelectorAll('.param-radio-group').forEach(el => el.disabled = true);
                    modelData.forEach(data => {
                        if (data.layout) {
                            modelParamsContainer.insertAdjacentHTML('beforeend', createModalParamsHTML(data.model, data.layout));
                        }
                    });
                } else {
                    modelParamsContainer.insertAdjacentHTML('beforeend', createModalParamsHTML(modelData[0].model, modelData[0].layout));
                }
            } else {
                console.error('No valid layout data available');
                modelParamsContainer.innerHTML = '<p>Unable to load model parameters</p>';
            }
        }
        
        console.log("Current Global Values after Update:", { selectedModels, isRegression, ensembleMethod, searchMethod });
        
        // Clear previous results
        clearResultsAndTables();
    }    

    function updateParamFields(model) {
        fetchModelLayout(model)
            .then(layout => {
                const modelParamsContainer = document.querySelector('.model-params');
                modelParamsContainer.innerHTML = '';
                modelParamsContainer.insertAdjacentHTML('beforeend', createModalParamsHTML(model, layout));
            })
            .catch(error => console.error("Error updating param fields:", error));
    }
    
    function getCurrentModelParams() {
        let params = {};
        document.querySelectorAll('.model-params input, .model-params select').forEach(element => {
            const name = element.getAttribute('name');
            const value = element.value;
            if (value !== '') {
                params[name] = value;
            }
        });
        console.log("Parameters being sent:", params);
        return params;
    }
    
    function clearResultsAndTables() {
        const bottomHalf = document.querySelector('.bottom-half');
        
        // Clear all result plots
        bottomHalf.querySelectorAll('.result-plot').forEach(plot => plot.src = '');
        
        // Clear score tables
        bottomHalf.querySelectorAll('#trainScoreTable tbody tr, #testScoreTable tbody tr').forEach(row => {
            row.querySelectorAll('td').forEach(cell => cell.textContent = '');
        });
        
        // Hide results block
        const resultsBlock = bottomHalf.querySelector('.results-block');
        if (resultsBlock) {
            resultsBlock.style.display = 'none';
        }
    }    
    
    function createModalParamsHTML(model, layout) {
        let html = `<div class="modal-params" data-model="${model}">`;
        for (const [key, value] of Object.entries(layout)) {
            html += `<div class="param-group">
                <label>${key}: </label>`;
            
            if (searchMethod === 'grid' || searchMethod === 'random') {
                html += `<input type="text" name="${model}-${key}" value="${value.value || ''}" placeholder="Enter comma-separated values">`;
            } else if (value.type === 'dropdown') {
                html += `<select name="${model}-${key}">`;
                value.options.forEach(option => {
                    html += `<option value="${option}" ${option === value.default ? 'selected' : ''}>${option}</option>`;
                });
                html += '</select>';
            } else {
                html += `<input type="${searchMethod === 'none' ? 'number' : 'text'}" name="${model}-${key}" value="${value.default || value.value}">`;
            }
            
            html += '</div>';
        }
        html += '</div>';
        return html;
    }
    
    function handleTrainPredictClick() {
        console.log("Train and Predict button clicked");
        const params = getCurrentModelParams();
    
        const modelType = isRegression ? "Regressor" : "Classifier";
        console.log(`${modelType}:`, selectedModels[0]);
        console.log("Ensemble Method:", ensembleMethod);
        console.log("Search Method:", searchMethod);
        console.log("Params:", params);
    
        fetch('/machinelearning/train_and_predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                models: selectedModels,
                model_type: modelType,
                ensemble_method: ensembleMethod,
                search_method: searchMethod,
                params: params
            }),
        })
        .then(response => response.json())
        .then(data => {
            console.log("Received response:", data);
            updateModalWithResults(data);
            isModelTrained = true;
        })
        .catch(error => {
            console.error("Error:", error);
        });
    }

    function updateModalWithResults(response) {
        console.log("Updating modal with results:", response);
        if (response.error) {
            console.error("Error in results:", response.error);
            alert("An error occurred: " + response.error);
            return;
        }
    
        // Update common elements
        updateScoreTable('#trainScoreTable', response.train_metrics);
        updateScoreTable('#testScoreTable', response.test_metrics);
        document.getElementById('bestParams').textContent = JSON.stringify(response.best_params, null, 2);
    
        // Update plots based on response type
        const mainPlot = document.getElementById('mainPlot');
        const additionalPlot1 = document.getElementById('additionalPlot1');
        const additionalPlot2 = document.getElementById('additionalPlot2');
    
        if (response.plot_url) {
            // Regression result
            mainPlot.src = 'data:image/png;base64,' + response.plot_url;
            additionalPlot1.style.display = 'none';
            additionalPlot2.style.display = 'none';
        } else if (response.confusion_matrix_plot) {
            // Classification result
            mainPlot.src = 'data:image/png;base64,' + response.confusion_matrix_plot;
            additionalPlot1.src = 'data:image/png;base64,' + response.roc_curve_plot;
            additionalPlot2.src = 'data:image/png;base64,' + response.learning_curve_plot;
            additionalPlot1.style.display = 'block';
            additionalPlot2.style.display = 'block';
        }
    
        document.querySelector('.results-block').style.display = 'block';
    }

    function updateModalWithResults(response) {
        console.log("Updating modal with results:", response);
        if (response.error) {
            console.error("Error in results:", response.error);
            alert("An error occurred: " + response.error);
            return;
        }
    
        // Update common elements
        updateScoreTable('#trainScoreTable', response.train_metrics);
        updateScoreTable('#testScoreTable', response.test_metrics);
        document.getElementById('bestParams').textContent = JSON.stringify(response.best_params, null, 2);
    
        // Update plots based on response type
        const mainPlot = document.getElementById('mainPlot');
        const additionalPlot1 = document.getElementById('additionalPlot1');
        const additionalPlot2 = document.getElementById('additionalPlot2');
    
        if (response.plot_url) {
            // Regression result
            mainPlot.src = 'data:image/png;base64,' + response.plot_url;
            additionalPlot1.style.display = 'none';
            additionalPlot2.style.display = 'none';
        } else if (response.confusion_matrix_plot) {
            // Classification result
            mainPlot.src = 'data:image/png;base64,' + response.confusion_matrix_plot;
            additionalPlot1.src = 'data:image/png;base64,' + response.roc_curve_plot;
            additionalPlot2.src = 'data:image/png;base64,' + response.learning_curve_plot;
            additionalPlot1.style.display = 'block';
            additionalPlot2.style.display = 'block';
        }
    
        document.querySelector('.results-block').style.display = 'block';
    }    
    
    function updateScoreTable(tableId, scores) {
        console.log(`Updating ${tableId} with scores:`, scores);
        const tbody = document.querySelector(`${tableId} tbody`);
        const rows = tbody.querySelectorAll('tr');
    
        scores.forEach((score, index) => {
            if (index < rows.length) {
                console.log(`Setting row ${index}:`, score);
                rows[index].querySelector('td:first-child').textContent = score[0];
                rows[index].querySelector('td:last-child').textContent = score[1];
            }
        });
    
        // Clear any remaining placeholder rows
        for (let i = scores.length; i < rows.length; i++) {
            rows[i].querySelectorAll('td').forEach(td => td.textContent = '');
        }
    }
    
    // SAVE TRAINED MODEL ///////////////////////////////////////////////////////////////////////////////////
    function handleSaveTrainedModel() {
        fetch('/machinelearning/download_trained_model')
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.blob();
            })
            .then(blob => {
                const link = document.createElement('a');
                link.href = window.URL.createObjectURL(blob);
                link.download = "trained_model.joblib";
                link.click();
                isModelSaved = true;
            })
            .catch(error => {
                console.error("Error downloading model:", error);
            });
    }    
    
    // SAVE DISPLAYED RESULT ///////////////////////////////////////////////////////////////////
    function handleSaveDisplayedResult() {
        const trainScoreData = getTableData('#trainScoreTable');
        const testScoreData = getTableData('#testScoreTable');
        const images = {};
        
        document.querySelectorAll('.result-plot').forEach(plot => {
            if (plot.src) {
                images[plot.id] = plot.src.split(',')[1];
            }
        });
    
        fetch('/machinelearning/save_displayed_result', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                trainScore: trainScoreData,
                testScore: testScoreData,
                images: images
            })
        })
        .then(response => response.blob())
        .then(blob => {
            const link = document.createElement('a');
            link.href = window.URL.createObjectURL(blob);
            link.download = 'model_results.zip';
            link.click();
        });
    }    
    
    function getTableData(tableId) {
        const data = [];
        document.querySelectorAll(`${tableId} tr`).forEach(row => {
            const rowData = [];
            row.querySelectorAll('td').forEach(cell => {
                rowData.push(cell.textContent);
            });
            if (rowData.length > 0) {
                data.push(rowData);
            }
        });
        return data;
    }
    
    // BEST PARAMS ///////////////////////////////////////////////////////////////////////////
    function handleViewOptimalParams() {
        const bestParamsContainer = document.getElementById('bestParamsContainer');
        if (bestParamsContainer.style.display !== 'none') {
            bestParamsContainer.style.display = 'none';
        } else {
            bestParamsContainer.style.display = 'block';
            bestParamsContainer.scrollIntoView({ behavior: 'smooth' });
        }
    }

    // EXPLAIN SHAP / LIME ////////////////////////////////////////////////////////////////////////////
    function showExplanationPlot(plotId, plotData) {
        const plotHTML = `
            <div id="${plotId}Container" class="explanation-plot">
                <div class="plot-header">
                    <h3>${plotId} Explanation</h3>
                    <div>
                        <button class="save-plot">Save</button>
                        <button class="close-plot">Close</button>
                    </div>
                </div>
                <img src="data:image/png;base64,${plotData.plot_url}" alt="${plotId} Explanation">
            </div>`;
    
        document.querySelector('.results-block').insertAdjacentHTML('beforeend', plotHTML);
        const plotContainer = document.getElementById(`${plotId}Container`);
    
        plotContainer.querySelector('.close-plot').addEventListener('click', () => plotContainer.remove());
        plotContainer.querySelector('.save-plot').addEventListener('click', () => downloadExplanationPlot(plotData.filename));
        plotContainer.scrollIntoView({ behavior: 'smooth' });
    }
    
    function downloadExplanationPlot(filename) {
        fetch('/machinelearning/download_explanation_plot', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ filename: filename })
        })
        .then(response => response.blob())
        .then(blob => {
            const link = document.createElement('a');
            link.href = window.URL.createObjectURL(blob);
            link.download = filename;
            link.click();
        });
    }
    
    function handleExplainShap() {
        fetch('/machinelearning/explain_shap', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                models: selectedModels,
                ensemble_method: ensembleMethod
            })
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(data => {
                    throw new Error(data.error || 'SHAP explanation failed');
                });
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                // Display error message to user
                const bottomHalf = document.querySelector('.bottom-half');
                const errorDiv = document.createElement('div');
                errorDiv.className = 'error-message';
                errorDiv.textContent = `Error: ${data.error}`;
                bottomHalf.appendChild(errorDiv);
            } else {
                showExplanationPlot('shapPlot', data);
            }
        })
        .catch(error => {
            console.error('SHAP Error:', error.message);
            // Display error message to user
            const bottomHalf = document.querySelector('.bottom-half');
            const errorDiv = document.createElement('div');
            errorDiv.className = 'error-message';
            errorDiv.textContent = `Error: ${error.message}`;
            bottomHalf.appendChild(errorDiv);
        });
    }    
    
    function handleExplainLime() {
        fetch('/machinelearning/explain_lime', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                models: selectedModels,
                model_type: isRegression ? "Regressor" : "Classifier",
                ensemble_method: ensembleMethod,
                search_method: searchMethod,
            })
        })
        .then(response => response.json())
        .then(data => showExplanationPlot('limePlot', data));
    }
    // HELP ////////////////////////////////////////////////////////////////////////////////////////
    function handleHelpButton() {
        const regressor = selectedModels[0];
        fetch(`/machinelearning/get_help?regressor=${regressor}`)
            .then(response => response.json())
            .then(data => showHelpModal(data.help_text, regressor));
    }
    
    function showHelpModal(helpText, regressor) {
        const modalHTML = `
            <div class="modal fade" id="helpModal" tabindex="-1" role="dialog">
                <div class="modal-dialog" role="document">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title">Parameter Tips for ${regressor}</h5>
                            <button type="button" class="close" data-dismiss="modal">&times;</button>
                        </div>
                        <div class="modal-body">
                            <pre style="color: blue;">${helpText}</pre>
                        </div>
                    </div>
                </div>
            </div>
        `;
        document.body.insertAdjacentHTML('beforeend', modalHTML);
        const modal = new bootstrap.Modal(document.getElementById('helpModal'));
        modal.show();
    }

    // CLOSE ////////////////////////////////////////////////////////////////////////////////////////////
    function handleCloseButton() {
        if (isModelTrained && !isModelSaved) {
            showCustomDialog(
                "You have not saved your trained model.",
                ["Save", "Don't Save", "Cancel"],
                handleCloseDialogChoice
            );
        } else {
            resetModalState();
        }
    }
    
    function handleCloseDialogChoice(choice) {
        if (choice === "Save") {
            saveTrainedModel(resetModalState);
        } else if (choice === "Don't Save") {
            resetModalState();
        }
        // If Cancel, do nothing and return to the modal
    }
    
    function showCustomDialog(message, buttons, callback) {
        const overlay = document.createElement('div');
        overlay.className = 'dialog-overlay';
        
        const dialog = document.createElement('div');
        dialog.className = 'custom-dialog';
        
        const messageElement = document.createElement('p');
        messageElement.textContent = message;
        dialog.appendChild(messageElement);
        
        const buttonContainer = document.createElement('div');
        buttonContainer.className = 'dialog-buttons';
        
        buttons.forEach(buttonText => {
            const button = document.createElement('button');
            button.textContent = buttonText;
            button.addEventListener('click', () => {
                overlay.remove();
                callback(buttonText);
            });
            buttonContainer.appendChild(button);
        });
        
        dialog.appendChild(buttonContainer);
        overlay.appendChild(dialog);
        document.body.appendChild(overlay);
        overlay.style.display = 'block';
    }
    
    function resetModalState() {
        // Reset model selection
        document.querySelectorAll('.button-group button').forEach(btn => btn.classList.remove('selected'));
        selectedModels = [];
    
        // Clear data containers
        document.getElementById('modelParametersContainer').innerHTML = '';
        document.querySelector('.results-block').style.display = 'none';
        document.getElementById('bestParamsContainer').style.display = 'none';
        document.querySelectorAll('.result-plot').forEach(plot => plot.src = '');
        document.querySelectorAll('#trainScoreTable tbody tr, #testScoreTable tbody tr td').forEach(cell => cell.textContent = '');
        document.getElementById('bestParams').textContent = '';
    
        // Reset radio buttons
        document.querySelector('input[name="ensemble_method"][value="none"]').checked = true;
        document.querySelector('input[name="search_method"][value="none"]').checked = true;
    
        // Reset state variables
        isModelTrained = false;
        isModelSaved = false;
        ensembleMethod = 'none';
        searchMethod = 'none';
        
        // Reset maxModels
        resetMaxModels();
    
        // Clear explanation plots
        ['shapPlotContainer', 'limePlotContainer'].forEach(id => {
            const container = document.getElementById(id);
            if (container) container.innerHTML = '';
        });
    
        // Hide modals
        ['regressionModal', 'classificationModal'].forEach(id => {
            const modal = bootstrap.Modal.getInstance(document.getElementById(id));
            if (modal) modal.hide();
        });
    
        // Clear modal content
        document.querySelector('.model-params').innerHTML = '';
        document.querySelector('.results-block').style.display = 'none';
        document.querySelectorAll('.tables-block table tbody td').forEach(cell => cell.textContent = '');
    }

})();
