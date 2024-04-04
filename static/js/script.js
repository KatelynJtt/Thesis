// Get page elements 

const fileForm = document.getElementById('fileForm');
const fileInput = document.getElementById('fileInput');

const uploadMessage = document.getElementById('uploadMessage');
const loadingIndicator = document.getElementById('loadingIndicator');

const uniPlot = document.getElementById('univariatePlot');
const biPlot = document.getElementById('bivariatePlot');
const multiPlot = document.getElementById('multivariatePlot');

// Submit form

fileForm.addEventListener('submit', async (e) => {

    e.preventDefault(); // This line prevents the page from refreshing
    // Show loading indicator
    loadingIndicator.style.display = 'block';
    
    // Get file 
    const file = fileInput.files[0];
    // Check if file is a CSV
    if (file.type !== 'text/csv') {
        uploadMessage.textContent = 'Please upload a CSV file.';
        return;
    } 
    // Create new FormData instance
    let formData = new FormData();
    // Append file to formData
    formData.append('file', file);

    // Show loading indicator
    loadingIndicator.style.display = 'block';

    // Submit file to Flask backend
    const response = await fetch('/upload', {
        method: 'POST',
        body: formData
    });

    // Hide loading indicator
    loadingIndicator.style.display = 'none';

    // Check if upload was successful
    if (response.ok) {

        uploadMessage.textContent = 'Upload successful!';
    } else {
        const responseBody = await response.json();
        console.log('Error:', responseBody.error);
        uploadMessage.textContent = 'Upload failed.';
        }
    
    // Show loading message by loading chunks
    if (!response.body) {
        throw Error("ReadableStream not yet supported in this browser.");
    }
    
    const reader = response.body.getReader();
    let {done, value} = await reader.read();
    let receivedLength = 0;  // received that many bytes at the moment
    let chunks = []; // array of received binary chunks (comprises the body)
    
    while (!done) {
        chunks.push(value);
        receivedLength += value.length;
        console.log(`Received ${receivedLength} bytes of data so far`);
        uploadMessage.textContent = `Creating table... ${Math.round(receivedLength / file.size * 100)}%`;
        ({done, value} = await reader.read());
    }
    
    let chunksAll = new Uint8Array(receivedLength); // (4.1)
    let position = 0;
    for(let chunk of chunks) {
        chunksAll.set(chunk, position); // (4.2)
        position += chunk.length;
    }
    
    let result = new TextDecoder("utf-8").decode(chunksAll);
    
    // Parse response and display results
    const {dataset, summary} = JSON.parse(result);
    displayDataset(dataset);
    displaySummary(summary);
  
});

//-------------------------------------------------##Display functions##

function displayDataset(data) {
    // Create table element
    const table = document.createElement('table');
    // Add header row
    const headerRow = document.createElement('tr');
    const headers = Object.keys(data[0]);

    headers.forEach(header => {
        const th = document.createElement('th');
        th.textContent = header;
        headerRow.appendChild(th);
    });

    table.appendChild(headerRow);
    // Add data rows
    data.forEach(row => {
        const tr = document.createElement('tr');
    
        headers.forEach(header => {
            const td = document.createElement('td');
            td.textContent = row[header];
            tr.appendChild(td);
        });
    
        table.appendChild(tr);
    });
    // Insert table into DOM
    let datasetDiv = document.getElementById('dataset');
    // Clear any existing content
    while (datasetDiv.firstChild) {
        datasetDiv.removeChild(datasetDiv.firstChild);
    }
    datasetDiv.appendChild(table);
}

function displaySummary(summary) {
    // Create table
    const table = document.createElement('table');
    // Add header row
    const headerRow = document.createElement('tr');
    const th = document.createElement('th');
    th.textContent = 'Statistic';
    headerRow.appendChild(th);

    const metricKeys = ['min', 'max', 'mean', 'median'];
    metricKeys.forEach(metric => {
        const th = document.createElement('th');
        th.textContent = metric;
        headerRow.appendChild(th);
    });

    table.appendChild(headerRow);
    // Add data rows
    Object.keys(summary).forEach(header => {
        const tr = document.createElement('tr');
        const th = document.createElement('th');
        th.textContent = header;
        tr.appendChild(th);

        metricKeys.forEach(metric => {
            const td = document.createElement('td');
            td.textContent = summary[header][metric];
            tr.appendChild(td);
        });

        table.appendChild(tr);
    });
    // Insert summary into DOM
    let summaryDiv = document.getElementById('summary');
    // Clear any existing content
    while (summaryDiv.firstChild) {
        summaryDiv.removeChild(summaryDiv.firstChild);
    }
    summaryDiv.appendChild(table);
}

function displayTable(dataset) {
    // Create table
    let table = document.createElement('table');

    // Add table header
    let thead = document.createElement('thead');
    let headers = Object.keys(dataset[0]);
    let headerRow = document.createElement('tr');
    headers.forEach(header => {
        let th = document.createElement('th');
        th.textContent = header;
        headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);
    table.appendChild(thead);

    // Add table body
    let tbody = document.createElement('tbody');
    dataset.forEach(row => {
        let tr = document.createElement('tr');
        headers.forEach(header => {
            let td = document.createElement('td');
            td.textContent = row[header];
            tr.appendChild(td);
        });
        tbody.appendChild(tr);
    });
    table.appendChild(tbody);

    // Insert table into DOM
    let tableDiv = document.getElementById('tableDiv');
    // Clear any existing content
    while (tableDiv.firstChild) {
        tableDiv.removeChild(tableDiv.firstChild);
    }
    tableDiv.appendChild(table);
}

//--------------------------------------------------##Plotting functions##

function plotUnivariate(data, header) {
    // Extract column data
    const column = data.map(row => row[header]);
    // Create trace
    const trace = {
        x: column,
        type: 'histogram',
    };
    // Define layout
    const layout = {
        title: `Distribution of ${header}`,
        xaxis: { title: header },
        yaxis: { title: 'Count' }
    };
    // Plot
    Plotly.newPlot(uniPlot, [trace], layout);
    }

    function plotBivariate(data, x, y) {
    // Extract column data
    const xData = data.map(row => row[x]);
    const yData = data.map(row => row[y]);
    // Create trace 
    const trace = {
        x: xData,
        y: yData,
        mode: 'markers'
    };
    // Define layout
    const layout = {
        title: `${x} vs ${y}`,
        xaxis: {title: x},
        yaxis: {title: y}
    };
    // Plot 
    Plotly.newPlot(biPlot, [trace], layout);
}

// Save image
async function saveImage(plotDiv) {
    // Get image data from Plotly
    const imageData = await Plotly.toImage(plotDiv);
    
    // Create download link
    const downloadLink = document.createElement('a');
    downloadLink.download = 'plot.png';

    // Convert image data to Blob
    const imageBlob = new Blob([imageData], {type: 'image/png'});

    // Set link href to object URL of Blob
    downloadLink.href = URL.createObjectURL(imageBlob);

    // Append link to DOM and trigger click
    document.body.appendChild(downloadLink);
    downloadLink.click();

    // Remove link and revoke object URL
    document.body.removeChild(downloadLink);
    URL.revokeObjectURL(downloadLink.href);
}
