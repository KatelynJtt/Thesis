// geo_imaging.js


// Otsu's method implementation in JavaScript
function otsu(data) {
    const hist = calculateHistogram(data, 100);
    const totalFreq = hist.reduce((sum, value) => sum + value, 0);
    const probabilities = hist.map(value => value / totalFreq);
  
    let maxVariance = 0;
    let optimalThreshold = 0;
  
    for (let t = 1; t < hist.length; t++) {
      const w0 = probabilities.slice(0, t).reduce((sum, value) => sum + value, 0);
      const w1 = probabilities.slice(t).reduce((sum, value) => sum + value, 0);
  
      if (w0 === 0 || w1 === 0) {
        continue;
      }
  
      const mean0 = probabilities.slice(0, t).reduce((sum, value, index) => sum + value * (index + 1), 0) / w0;
      const mean1 = probabilities.slice(t).reduce((sum, value, index) => sum + value * (index + t + 1), 0) / w1;
  
      const variance = w0 * w1 * (mean0 - mean1) ** 2;
  
      if (variance > maxVariance) {
        maxVariance = variance;
        optimalThreshold = t;
      }
    }
  
    return optimalThreshold;
  }
  
  // Helper function to calculate histogram
  function calculateHistogram(data, numBins) {
    const min = Math.min(...data);
    const max = Math.max(...data);
    const binWidth = (max - min) / numBins;
    const hist = new Array(numBins).fill(0);
  
    for (const value of data) {
      const binIndex = Math.floor((value - min) / binWidth);
      hist[binIndex]++;
    }
  
    return hist;
  }


// Function to plot the vegetation index histogram
function plot_VI_hist(ax, title, data, plotIndex) {
    
    //ax: An object representing the plot area where the histogram will be rendered.
    //title: The title of the vegetation index.
    //data: A 1D array containing the vegetation index values.
    //plotIndex: The index of the plot being visualized.

    const hist = calculateHistogram(data, 100);
    const bins = hist.map((value, index) => index);
    const optimalThreshold = otsu(data);
  
    // Plot the histogram
    const cmap = 'viridis';
    const colors = hist.map((value, index) => getColorFromColormap(cmap, index / hist.length));
    const barWidth = bins[1] - bins[0];
  
    if (title === 'ExR' || title === 'CIVE') {
      colors.reverse();
    }
  
    ax.data = [];
    for (let i = 0; i < hist.length; i++) {
      ax.data.push({
        x: [bins[i], bins[i] + barWidth],
        y: [0, hist[i]],
        type: 'rect',
        fillcolor: colors[i],
        line: { width: 0 }
      });
    }
  
    // Add colorbar
    const colorbar = {
      x: 1.05,
      xanchor: 'left',
      y: 0.5,
      yanchor: 'middle',
      len: 0.6,
      title: 'Vegetation Index Value',
      titleside: 'right'
    };
    ax.colorbar = colorbar;
  
    // Add title and labels
    ax.title = `Plot${plotIndex}_${title}<br>Th=${optimalThreshold.toFixed(2)}`;
    ax.xaxis = { title: 'Vegetation Index Value' };
    ax.yaxis = { title: 'Frequency' };
  
    // Render the plot
    Plotly.newPlot(ax.id, ax.data, ax.layout);
  }
  
  // Helper function to get color from a colormap
  function getColorFromColormap(cmap, value) {
    const colormap = {
      'viridis': [
        [0.267004, 0.004874, 0.329415],
        [0.275243, 0.009525, 0.346235],
        // ... (more colors in the 'viridis' colormap)
      ]
    };
  
    const colors = colormap[cmap];
    const colorIndex = Math.floor(value * (colors.length - 1));
    return `rgb(${colors[colorIndex].map(c => Math.floor(c * 255)).join(',')})`;
  }



// Function to handle GeoTIFF action
function handleGeoTIFF(dataset) {
    dataset.clip_rasterio_shape();
    // Add any additional functionality you need for GeoTIFF
  }
  
  // Function to handle VI Histogram Visualization action
  function handleVIHistogramVisualization(dataset, plotIndex, isHist) {
    dataset.visualization_plot(plotIndex, isHist);
    // Add any additional functionality you need for VI Histogram Visualization
  }
  
  // Function to handle Clip Image action
  function handleClipImage(dataset) {
    dataset.clip_rasterio_shape();
    // Add any additional functionality you need for Clip Image
  }
  
  // Function to handle Auto Dataset Extraction action
  function handleAutoDatasetExtraction(dataset, targetDataFrame) {
    if (dataset instanceof RGB2Dataset) {
      dataset.dataset_extraction_auto(targetDataFrame);
    } else if (dataset instanceof MS2Dataset) {
      dataset.dataset_extraction_auto(targetDataFrame);
    }
    // Add any additional functionality you need for Auto Dataset Extraction
  }
  
  // Function to handle Manual Dataset Extraction action
  function handleManualDatasetExtraction(dataset) {
    if (dataset instanceof RGB2Dataset) {
      manu_extraction_window_rgb(dataset);
    } else if (dataset instanceof MS2Dataset) {
      manu_extraction_window_ms(dataset);
    }
    // Add any additional functionality you need for Manual Dataset Extraction
  }
  
  // Event listeners for button clicks
  document.addEventListener('DOMContentLoaded', function() {
    const geoTIFFButton = document.getElementById('geoTIFFButton');
    const viHistogramButton = document.getElementById('viHistogramButton');
    const clipImageButton = document.getElementById('clipImageButton');
    const autoExtractButton = document.getElementById('autoExtractButton');
    const manualExtractButton = document.getElementById('manualExtractButton');
  
    geoTIFFButton.addEventListener('click', function() {
      const dataset = createDataset(); // Implement this function to create the appropriate dataset instance
      handleGeoTIFF(dataset);
    });
  
    viHistogramButton.addEventListener('click', function() {
      const dataset = createDataset(); // Implement this function to create the appropriate dataset instance
      const plotIndex = prompt('Enter plot index:');
      const isHist = confirm('Do you want to visualize the histogram?');
      handleVIHistogramVisualization(dataset, plotIndex, isHist);
    });
  
    clipImageButton.addEventListener('click', function() {
      const dataset = createDataset(); // Implement this function to create the appropriate dataset instance
      handleClipImage(dataset);
    });
  
    autoExtractButton.addEventListener('click', function() {
      const dataset = createDataset(); // Implement this function to create the appropriate dataset instance
      const targetDataFrame = getTargetDataFrame(); // Implement this function to get the target data frame
      handleAutoDatasetExtraction(dataset, targetDataFrame);
    });
  
    manualExtractButton.addEventListener('click', function() {
      const dataset = createDataset(); // Implement this function to create the appropriate dataset instance
      handleManualDatasetExtraction(dataset);
    });
  });
  