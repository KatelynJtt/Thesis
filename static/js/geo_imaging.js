function handleRGBManualExtraction() {
    const rgbManualExtractionSection = document.getElementById('rgb-manual-extraction');
    const rgbViSelect = document.getElementById('rgb-vi-select');
    const rgbManualExtractionPlots = document.getElementById('rgb-manual-extraction-plots');
    const rgbManualExtractionOk = document.getElementById('rgb-manual-extraction-ok');

    rgbManualExtractionSection.style.display = 'block';

    // Clear previous plots
    rgbManualExtractionPlots.innerHTML = '';

    // Fetch data from Flask route
    fetch('/rgb-manual-extraction', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ selectedVI: rgbViSelect.value })
    })
    .then(response => response.json())
    .then(data => {
        // Update the plots section with the received data
        data.plots.forEach(plot => {
            const plotDiv = document.createElement('div');
            plotDiv.innerHTML = plot;
            rgbManualExtractionPlots.appendChild(plotDiv);
        });
    })
    .catch(error => {
        console.error('Error:', error);
    });

    rgbManualExtractionOk.addEventListener('click', () => {
        // Handle OK button click
        rgbManualExtractionSection.style.display = 'none';
        // Perform any additional actions if needed
    });
}