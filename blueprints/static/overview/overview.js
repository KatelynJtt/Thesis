(function() {
    let currentFile = null;

    window.initializeOverview = function() {
        console.log('Overview script loaded');
        const filename = getSelectedFile();
        
        if (filename) {
            currentFile = filename;
            loadFileData(currentFile);
            enableOverviewFunctions();
        } else {
            displayMessage('Please select a file from the source column to use Overview functions.');
            disableOverviewFunctions();
        }
    };

    function loadFileData(filename) {
        console.log('Attempting to load file:', filename);
        fetch(`/overview/process_file?file=${filename}`)
            .then(response => {
                console.log('Response status:', response.status);
                return response.json();
            })
            .then(data => {
                console.log('Received data:', data);
                if (data.success) {
                    displayOverviewData(data.data);
                }
            });
    }    

    function displayOverviewData(data) {
        // Update summary stats
        document.querySelector('.summary-stats').innerHTML = `
            <p>
                Total Rows: ${data.row_count} | 
                Total Columns: ${data.column_count} | 
                Memory Usage: ${data.memory_usage} | 
                Missing Values: ${data.missing_count} | 
                Duplicate Rows: ${data.duplicate_count}
            </p>
        `;
        
        // Update data preview
        document.querySelector('.sample-data').innerHTML = `
            <h4>First 5 Rows</h4>
            ${data.head_data}
            <h4>Random Sample (5 Rows)</h4>
            ${data.sample_data}
        `;
        
        // Setup column selector with event handling
        const columnSelector = document.getElementById('column-selector');
        columnSelector.innerHTML = '<option value="All">All Columns</option>';
        data.columns.forEach(column => {
            columnSelector.innerHTML += `<option value="${column}">${column}</option>`;
        });
    
        // Add event listener for column selection
        columnSelector.addEventListener('change', function(e) {
            const column = e.target.value;
            updateColumnDetails(column);
        });
    }

    function updateColumnDetails(column) {
        fetch(`/overview/column-details/${column}?file=${currentFile}`)
            .then(response => response.json())
            .then(data => {
                const detailsDiv = document.getElementById('column-details');
                if (data.type === 'all') {
                    detailsDiv.innerHTML = `
                        <h4>Dataset Overview</h4>
                        <p>Total Columns: ${data.details.total_columns}</p>
                        <p>Numeric Columns: ${data.details.numeric_columns}</p>
                        <p>Categorical Columns: ${data.details.categorical_columns}</p>
                    `;
                } else if (data.type === 'numeric') {
                    detailsDiv.innerHTML = `
                        <h4>${column} Details</h4>
                        <p>Type: Numeric</p>
                        <p>Missing Values: ${data.missing_values}</p>
                        <p>Unique Values: ${data.unique_values}</p>
                        <div class="stats-container">
                            ${statSections.map(section => `
                                <div class="stat-section">
                                    <h5 class="section-title">${section.title}</h5>
                                    <div class="stats-row">
                                        ${section.stats.map(stat => `
                                            <div class="stat-item" title="${statDescriptions[stat]}">
                                                <span class="stat-label">${stat.charAt(0).toUpperCase() + stat.slice(1)}:</span>
                                                <span class="stat-value">${data[stat].toFixed(2)}</span>
                                                <span class="info-icon">ℹ️</span>
                                            </div>
                                        `).join('')}
                                    </div>
                                </div>
                            `).join('')}
                        </div>
                    `;
                // If column is categorical
                } else {
                    // Check if all values are unique
                    if (data.unique_values === data.total_values) {
                        detailsDiv.innerHTML = `
                            <h4>${column} Details</h4>
                            <p>Type: Categorical</p>
                            <p>Missing Values: ${data.missing_values}</p>
                            <p>Unique Values: ${data.unique_values}</p>
                            <div class="unique-values-notice">
                                <h5>Value Distribution</h5>
                                <p>All values in this column are unique - each value appears exactly once.</p>
                            </div>
                        `;
                    } else {
                        detailsDiv.innerHTML = `
                            <h4>${column} Details</h4>
                            <p>Type: Categorical</p>
                            <p>Missing Values: ${data.missing_values}</p>
                            <p>Unique Values: ${data.unique_values}</p>
                            <h5>Top 5 Most Frequent Values</h5>
                            <table class="category-table">
                                <thead>
                                    <tr>
                                        <th>Value</th>
                                        <th>Frequency</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    ${Object.entries(data.value_counts)
                                        .map(([value, count]) => `
                                            <tr>
                                                <td>${value}</td>
                                                <td>${count}</td>
                                            </tr>
                                        `).join('')}
                                </tbody>
                            </table>
                        `;
                    }
                }

                // Add column values display
                const valuesDiv = document.getElementById('column-values');
                fetch(`/overview/column-values/${column}?file=${currentFile}`)
                    .then(response => response.json())
                    .then(valuesData => {
                        valuesDiv.innerHTML = `
                            <div class="values-header">
                                <h4>${column} Values</h4>
                            </div>
                            <div class="values-scroll">
                                ${valuesData.values.map(value => `<div>${value}</div>`).join('')}
                            </div>
                        `;
                    });
                
                // After updating the content, set the height of values column
                const detailsHeight = document.getElementById('column-details').offsetHeight;
                document.getElementById('column-values').style.height = `${detailsHeight}px`;
            });
    }
    
    const statDescriptions = {
        mean: "Average of all values. Sensitive to outliers.",
        median: "Middle value when sorted. More robust to outliers than mean.",
        std: "Standard Deviation - Measures spread of data around the mean.",
        min: "Smallest value in the dataset.",
        max: "Largest value in the dataset.",
        skewness: "Measures asymmetry of distribution. 0 indicates symmetry.",
        kurtosis: "Measures 'tailedness' of distribution. Higher values indicate more outliers.",
        q1: "First quartile - 25% of data falls below this value.",
        q3: "Third quartile - 75% of data falls below this value.",
        iqr: "Interquartile Range - Range between Q1 and Q3, used for outlier detection.",
        mode: "Most frequent value in the dataset.",
        variance: "Average squared deviation from mean. Larger values indicate greater spread."
    };
    
    // Define sections with titles and their stats
    const statSections = [
        {
            title: "Range Statistics",
            stats: ['min', 'max']
        },
        {
            title: "Central Tendency",
            stats: ['mean', 'median']
        },
        {
            title: "Distribution Quartiles",
            stats: ['q1', 'q3']
        },
        {
            title: "Spread Measures",
            stats: ['std', 'variance']
        },
        {
            title: "Distribution Shape",
            stats: ['skewness', 'kurtosis']
        },
        {
            title: "Additional Measures",
            stats: ['mode', 'iqr']
        }
    ];

    // Call initialization when DOM is ready
    initializeOverview();

    function getSelectedFile() {
        const selectedFileBtn = document.querySelector('.file-btn.selected');
        return selectedFileBtn ? selectedFileBtn.dataset.filename : null;
    }

    function displayMessage(message, isError = false) {
        const messageElement = document.getElementById('overview-message') || document.createElement('div');
        messageElement.id = 'overview-message';
        messageElement.textContent = message;
        messageElement.style.color = isError ? 'red' : 'green';
        messageElement.style.padding = '10px';
        messageElement.style.marginBottom = '20px';
        const overviewContent = document.getElementById('overview-content');
        overviewContent.insertBefore(messageElement, overviewContent.firstChild);
    }

    function disableOverviewFunctions() {
        document.querySelectorAll('#overview-content button, #overview-content select').forEach(el => el.disabled = true);
    }

    function enableOverviewFunctions() {
        document.querySelectorAll('#overview-content button, #overview-content select').forEach(el => el.disabled = false);
    }

    function processSelectedFile() {
        const filename = getSelectedFile();
        if (filename) {
            fetch(`/overview/process_file?file=${filename}`)
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        displayAllColumns();
                    } else {
                        displayMessage('Error: ' + data.error, true);
                    }
                });
        }
    }

    function displayAllColumns() {
        const filename = getSelectedFile();
        fetch(`/overview/all-columns/${filename}`)
            .then(response => response.json())
            .then(data => {
                const detailsDiv = document.getElementById('column-details');
                detailsDiv.innerHTML = generateColumnSummaryHTML(data);
            });
    }

    function fetchColumnDetails(column) {
        const filename = getSelectedFile();
        fetch(`/overview/column-details/${filename}/${column}`)
            .then(response => response.json())
            .then(data => {
                const detailsDiv = document.getElementById('column-details');
                if (data.type === 'numerical') {
                    detailsDiv.innerHTML = generateNumericalDetailsHTML(data, column);
                    createDistributionPlot(data.distribution, column);
                } else {
                    detailsDiv.innerHTML = generateCategoricalDetailsHTML(data, column);
                    createBarPlot(data.value_counts, column);
                }
            });
    }

    function generateNumericalDetailsHTML(data, column) {
        return `
            <div class="column-stats">
                <h4>${column} (Numerical)</h4>
                <p>Mean: ${data.mean.toFixed(2)}</p>
                <p>Median: ${data.median.toFixed(2)}</p>
                <p>Standard Deviation: ${data.std.toFixed(2)}</p>
                <p>Min: ${data.min}</p>
                <p>Max: ${data.max}</p>
                <p>Missing Values: ${data.null_count}</p>
                <p>Quartiles:</p>
                <ul>
                    <li>25%: ${data.quartiles[0].toFixed(2)}</li>
                    <li>50%: ${data.quartiles[1].toFixed(2)}</li>
                    <li>75%: ${data.quartiles[2].toFixed(2)}</li>
                </ul>
                <div id="distribution-plot"></div>
            </div>
        `;
    }

    function generateCategoricalDetailsHTML(data, column) {
        return `
            <div class="column-stats">
                <h4>${column} (Categorical)</h4>
                <p>Unique Values: ${data.unique_values}</p>
                <p>Missing Values: ${data.null_count}</p>
                <p>Top 5 Values:</p>
                <div id="category-plot"></div>
            </div>
        `;
    }

    function createDistributionPlot(distribution, column) {
        const trace = {
            x: distribution,
            type: 'histogram',
            name: column
        };
        
        const layout = {
            title: `Distribution of ${column}`,
            xaxis: { title: column },
            yaxis: { title: 'Count' }
        };

        Plotly.newPlot('distribution-plot', [trace], layout);
    }

    function createBarPlot(valueCounts, column) {
        const trace = {
            x: Object.keys(valueCounts),
            y: Object.values(valueCounts),
            type: 'bar',
            name: column
        };
        
        const layout = {
            title: `Value Counts for ${column}`,
            xaxis: { title: 'Categories' },
            yaxis: { title: 'Count' }
        };

        Plotly.newPlot('category-plot', [trace], layout);
    }

})();
