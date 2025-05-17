var MLApp = MLApp || {};

// Check for stored values and use them if available, otherwise use default values
MLApp.selectedModels = window.MLAppTempStorage.selectedModels || [];
MLApp.maxModels = window.MLAppTempStorage.maxModels || 1;
MLApp.modelParams = window.MLAppTempStorage.modelParams || {};
MLApp.isRegression = window.MLAppTempStorage.isRegression !== undefined ? window.MLAppTempStorage.isRegression : false;
MLApp.ensembleMethod = window.MLAppTempStorage.ensembleMethod || 'none';
MLApp.searchMethod = window.MLAppTempStorage.searchMethod || 'none';
MLApp.modalId = window.MLAppTempStorage.modalId || '';
MLApp.isModelTrained = window.MLAppTempStorage.isModelTrained || false;
MLApp.isModelSaved = window.MLAppTempStorage.isModelSaved || false;

// Clear the temporary storage after using it
window.MLAppTempStorage = {};

// Log the updated MLApp object
console.log("Updated MLApp object:", JSON.stringify(MLApp, null, 2));
