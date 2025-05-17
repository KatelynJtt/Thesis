$(document).ready(function() {

/////Functions added to MLApp for global use //////////////////////////////////////////////////////////////////////////
    // SHOW MODAL WINDOWS - based on model type (regressor / classifier)
    MLApp.showModelModal = function(models, regression, ensemble) {
        console.log("Start of showModelModal. Received ensemble:", ensemble);
        
        MLApp.selectedModels = models;
        MLApp.isRegression = regression;
        MLApp.ensembleMethod = ensemble;
        MLApp.searchMethod = $('input[name="search_method"]:checked').val() || 'none';
        MLApp.modalId = MLApp.isRegression ? '#regressionModal' : '#classificationModal';
    
        console.log("After setting values in showModelModal. Ensemble method:", MLApp.ensembleMethod);
        console.log("Global Values after show: " + JSON.stringify(MLApp));
    
        $(MLApp.modalId).modal('show');
        MLApp.updateModelLayouts();
    
        console.log("End of showModelModal. Final ensemble method:", MLApp.ensembleMethod);
    }    


    MLApp.updateModelLayouts = function() {
        Promise.all(MLApp.selectedModels.map(model => MLApp.fetchModelLayout(model)))
            .then(layouts => {
                const modelData = MLApp.selectedModels.map((model, index) => ({
                    model: model,
                    layout: layouts[index]
                }));
                MLApp.updateModalContent(modelData);
            })
            .catch(error => console.error("Error fetching model layouts:", error));
    }


    MLApp.updateModalContent = function(modelData) {
        // Set modal title and model name
        const modelTitle = MLApp.ensembleMethod !== 'none' ? MLApp.ensembleMethod : modelData[0].model;
        $(`${MLApp.modalId} .modal-title`).text(modelTitle + (MLApp.isRegression ? ' Regression' : ' Classification'));
        $(`${MLApp.modalId} #${MLApp.isRegression ? 'regressor' : 'classifier'}`).text(modelTitle);

        // Update model-params block
        const modelParamsContainer = $(`${MLApp.modalId} .model-params`);
        modelParamsContainer.empty();

        if (modelData && modelData.length > 0 && modelData[0].layout) {
            if (MLApp.ensembleMethod !== 'none') {
                $(`${MLApp.modalId} .param-radio-group`).prop('disabled', true);
                modelData.forEach(data => {
                    if (data.layout) {
                        modelParamsContainer.append(MLApp.createModalParamsHTML(data.model, data.layout));
                    }
                });
            } else {
                modelParamsContainer.append(MLApp.createModalParamsHTML(modelData[0].model, modelData[0].layout));
            }
        } else {
            console.error('No valid layout data available');
            modelParamsContainer.append('<p>Unable to load model parameters</p>');
        }
        console.log("Current Global Values after Update: " + JSON.stringify(MLApp));
        // Clear previous results
        MLApp.clearResultsAndTables();
    }

    MLApp.updateParamFields = function() {    
        MLApp.fetchModelLayout(model)
            .then(layout => {
                const modelParamsContainer = $('.model-params');
                modelParamsContainer.empty();
                modelParamsContainer.append(MLApp.createModalParamsHTML(model, layout));
            })
            .catch(error => console.error("Error updating param fields:", error));
    }

    MLApp.getCurrentModelParams = function() {
        let params = {};
        $('.model-params input, .model-params select').each(function() {
            const name = $(this).attr('name');
            const value = $(this).val();
            if (value !== '') {
                params[name] = value;
            }
        });
        console.log("Parameters being sent:", params);
        return params;
    }

    // CLEAR RESULTS AND TABLES
    MLApp.clearResultsAndTables = function() {
        // Clear all result plots
        $(`${MLApp.modalId} .result-plot`).attr('src', '');
        
        // Clear score tables
        $(`${MLApp.modalId} #trainScoreTable tbody tr, ${MLApp.modalId} #testScoreTable tbody tr`).each(function() {
            $(this).find('td').text('');
        });

        // Hide results block
        $(`${MLApp.modalId} .results-block`).hide();
    }

    // Create model param elements
    MLApp.createModalParamsHTML = function(model, layout) {
        let html = `<div class="modal-params" data-model="${model}">`;
        for (const [key, value] of Object.entries(layout)) {
            html += `<div class="param-group">
                <label>${key}: </label>`;
            
            if (MLApp.searchMethod === 'grid' || MLApp.searchMethod === 'random') {
                html += `<input type="text" name="${model}-${key}" value="${value.value || ''}" placeholder="Enter comma-separated values">`;
            } else if (value.type === 'dropdown') {
                html += `<select name="${model}-${key}">`;
                value.options.forEach(option => {
                    html += `<option value="${option}" ${option === value.default ? 'selected' : ''}>${option}</option>`;
                });
                html += '</select>';
            } else {
                html += `<input type="${MLApp.searchMethod === 'none' ? 'number' : 'text'}" name="${model}-${key}" value="${value.default || value.value}">`;
            }
            
            html += '</div>';
        }
        html += '</div>';
        return html;
    }

///////////////////////////////////////////////////////////////////////////////////////////////////////

    $('input[name="search_method"]').change(function() {
        MLApp.searchMethod = $(this).val();
        MLApp.updateModelLayouts();
    });

    // listen for other option
    $(document).on('change', '.param-select', function() {
        const customInput = $(this).siblings('.custom-input');
        if ($(this).val() === 'other') {
            customInput.show();
        } else {
            customInput.hide();
        }
    });

    // TRAIN AND PREDICT button click event
    $(document).on('click', '#trainPredict', function() {
        console.log("Train and Predict button clicked");
        const params = MLApp.getCurrentModelParams();
    
        const modelType = MLApp.isRegression ? "Regressor" : "Classifier";
        console.log(`${modelType}:`, MLApp.selectedModels[0]);
        console.log("Ensemble Method:", MLApp.ensembleMethod);
        console.log("Search Method:", MLApp.searchMethod);
        console.log("Params:", params);
    
        $.ajax({
            url: '/train_and_predict',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                models: MLApp.selectedModels,
                model_type: modelType,
                ensemble_method: MLApp.ensembleMethod,
                search_method: MLApp.searchMethod,
                params: params
            }),
            success: function(response) {
                console.log("Received response:", response);
                updateModalWithResults(response);
                MLApp.isModelTrained = true;
            },
            error: function(xhr, status, error) {
                console.error("Error:", error);
                console.log("Status:", status);
                console.log("Response:", xhr.responseText);
            }
        });
    });

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
        $('#bestParams').text(JSON.stringify(response.best_params, null, 2));
    
        // Update plots based on response type
        if (response.plot_url) {
            // Regression result
            $('#mainPlot').attr('src', 'data:image/png;base64,' + response.plot_url);
            $('#additionalPlot1, #additionalPlot2').hide();
        } else if (response.confusion_matrix_plot) {
            // Classification result
            $('#mainPlot').attr('src', 'data:image/png;base64,' + response.confusion_matrix_plot);
            $('#additionalPlot1').attr('src', 'data:image/png;base64,' + response.roc_curve_plot).show();
            $('#additionalPlot2').attr('src', 'data:image/png;base64,' + response.learning_curve_plot).show();
        }
    
        $('.results-block').show();
    }

    function updateScoreTable(tableId, scores) {
        console.log(`Updating ${tableId} with scores:`, scores);
        const tbody = $(tableId).find('tbody');
        const rows = tbody.find('tr');
        scores.forEach((score, index) => {
            if (index < rows.length) {
                console.log(`Setting row ${index}:`, score);
                $(rows[index]).find('td:first').text(score[0]);
                $(rows[index]).find('td:last').text(score[1]);
            }
        });
        // Clear any remaining placeholder rows
        for (let i = scores.length; i < rows.length; i++) {
            $(rows[i]).find('td').text('');
        }
    }    

    // SAVE TRAINED MODEL ///////////////////////////////////////////////////////////////////////////////////
    function saveTrainedModel(callback) {
        $.ajax({
            url: '/download_trained_model',
            type: 'GET',
            xhrFields: {
                responseType: 'blob'
            },
            success: function(blob) {
                var link = document.createElement('a');
                link.href = window.URL.createObjectURL(blob);
                link.download = "trained_model.joblib";
                link.click();
                MLApp.isModelSaved = true;
                if (callback) callback();
            },
            error: function(xhr, status, error) {
                console.error("Error downloading model:", error);
                alert("Error downloading model. Please try again.");
            }
        });
    }

    // SAVE DISPLAYED RESULT ///////////////////////////////////////////////////////////////////
    $(document).on('click', '#saveDisplayedResult', function(e) {
        e.preventDefault();
        
        let trainScoreData = getTableData('#trainScoreTable');
        let testScoreData = getTableData('#testScoreTable');
        
        let images = {};
        $('.result-plot').each(function() {
            let imgId = $(this).attr('id');
            let imgSrc = $(this).attr('src');
            if (imgSrc) {
                images[imgId] = imgSrc;
            }
        });
    
        $.ajax({
            url: '/save_displayed_result',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                trainScore: trainScoreData,
                testScore: testScoreData,
                images: images
            }),
            xhrFields: {
                responseType: 'blob'
            },
            success: function(blob) {
                var link = document.createElement('a');
                link.href = window.URL.createObjectURL(blob);
                link.download = 'model_results.zip';
                link.click();
            },
            error: function() {
                alert('Error saving results.');
            }
        });
    });           

    function getTableData(tableId) {
        let data = [];
        $(tableId + ' tr').each(function() {
            let row = [];
            $(this).find('td').each(function() {
                row.push($(this).text());
            });
            if (row.length > 0) {
                data.push(row);
            }
        });
        return data;
    }

    // BEST PARAMS ///////////////////////////////////////////////////////////////////////////
    $(document).on('click', '#viewOptimalParams', function() {
        var $bestParamsContainer = $('#bestParamsContainer');
        if ($bestParamsContainer.is(':visible')) {
            $bestParamsContainer.hide();
        } else {
            $bestParamsContainer.show();
            $('html, body').animate({
                scrollTop: $bestParamsContainer.offset().top
            }, 500);
        }
    });   

    // EXPLAIN SHAP / LIME ////////////////////////////////////////////////////////////////////////////
    function showExplanationPlot(plotId, plotData) {
        const plotContainer = $(`<div id="${plotId}Container" class="explanation-plot">
            <div class="plot-header">
                <h3>${plotId} Explanation</h3>
                <div>
                    <button class="save-plot">Save</button>
                    <button class="close-plot">Close</button>
                </div>
            </div>
            <img src="data:image/png;base64,${plotData.plot_url}" alt="${plotId} Explanation">
        </div>`);
        
        $('.results-block').append(plotContainer);
        
        plotContainer.find('.close-plot').click(function() {
            plotContainer.remove();
        });
        
        plotContainer.find('.save-plot').click(function() {
            $.ajax({
                url: '/download_explanation_plot',
                type: 'POST',
                data: JSON.stringify({ filename: plotData.filename }),
                contentType: 'application/json',
                xhrFields: {
                    responseType: 'blob'
                },
                success: function(blob) {
                    var link = document.createElement('a');
                    link.href = window.URL.createObjectURL(blob);
                    link.download = plotData.filename;
                    link.click();
                }
            });
        });
        
        $('html, body').animate({
            scrollTop: plotContainer.offset().top
        }, 1000);
    }

    $(document).on('click', '#explainShap', function() {
        $.ajax({
            url: '/explain_shap',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                models: MLApp.selectedModels,
                ensemble_method: MLApp.ensembleMethod
            }),
            success: function(response) {
                showExplanationPlot('shapPlot', response);
            }
        });
    });   

    $(document).on('click', '#explainLime', function() {
        $.ajax({
            url: '/explain_lime',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                models: MLApp.selectedModels,
                model_type: MLApp.isRegression ? "Regressor" : "Classifier",
                ensemble_method: MLApp.ensembleMethod,
                search_method: MLApp.searchMethod,
            }),
            success: function(response) {
                showExplanationPlot('limePlot', response);
            }
        });
    });        

    // HELP ////////////////////////////////////////////////////////////////////////////////////////
    $(document).on('click', '#helpButton', function() {
        regressor = MLApp.selectedModels[0]
        $.ajax({
            url: '/get_help',
            type: 'GET',
            data: { regressor: regressor },
            success: function(response) {
                showHelpModal(response.help_text, regressor);
            }
        });
    });

    function showHelpModal(helpText, regressor) {
        const modal = `
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
        $('body').append(modal);
        $('#helpModal').modal('show');
    }

    // CLOSE ////////////////////////////////////////////////////////////////////////////////////////////
    $(document).on('click', '#closeButton', function() {
        if (MLApp.isModelTrained && !MLApp.isModelSaved) {
            showCustomDialog("You have not saved your trained model.", 
                ["Save", "Don't Save", "Cancel"], 
                function(choice) {
                    if (choice === "Save") {
                        saveTrainedModel(function() {
                            resetModalState();
                        });
                    } else if (choice === "Don't Save") {
                        resetModalState();
                    }
                    // If Cancel, do nothing and return to the modal
                }
            );
        } else {
            resetModalState();
        }
    });
    
    function showCustomDialog(message, buttons, callback) {
        let overlay = $('<div class="dialog-overlay"></div>').appendTo('body');
        let dialog = $('<div class="custom-dialog"></div>').appendTo(overlay);
        $('<p></p>').text(message).appendTo(dialog);
        let buttonContainer = $('<div class="dialog-buttons"></div>').appendTo(dialog);
        buttons.forEach(function(buttonText) {
            $('<button></button>').text(buttonText).click(function() {
                overlay.remove();
                callback(buttonText);
            }).appendTo(buttonContainer);
        });
        overlay.show();
    }
    
    function resetModalState() {
        // Reset model selection
        $('.button-group button').removeClass('selected');
        MLApp.selectedModels = [];
    
        // Clear data containers
        $('#modelParametersContainer').empty();
        $('.results-block').hide();
        $('#bestParamsContainer').hide();
        $('.result-plot').attr('src', '');
        $('#trainScoreTable tbody tr, #testScoreTable tbody tr').find('td').text('');
        $('#bestParams').text('');
    
        // Reset radio buttons
        $('input[name="ensemble_method"][value="none"]').prop('checked', true);
        $('input[name="search_method"][value="none"]').prop('checked', true);
    
        // Reset state variables
        MLApp.isModelTrained = false;
        MLApp.isModelSaved = false;
        MLApp.ensembleMethod = 'none';
        MLApp.searchMethod = 'none';
        // Reset MLApp.maxModels
        resetMaxModels();
    
        // Clear any explanation plots
        $('#shapPlotContainer, #limePlotContainer').empty();
    
        // Hide the entire modal
        $('#regressionModal, #classificationModal').modal('hide');
    
        // Reset model count
        MLApp.resetMaxModels()
    
        // Clear the modal content
        $('.model-params').empty();
        $('.results-block').hide();
        $('.tables-block table tbody').find('td').text('');
    }    
});