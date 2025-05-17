document.addEventListener('DOMContentLoaded', function() {
    // Check script loaded
    console.log('Base Script loaded');

    const addSourceBtn = document.getElementById('addSourceBtn');
    const uploadModal = document.getElementById('uploadModal');
    const cancelBtn = document.getElementById('cancelBtn');
    const uploadBtn = document.getElementById('uploadBtn');
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const fileList = document.getElementById('fileList');

    const tabStates = {
        isHome: false,
        isPreprocessing: false,
        isEDA: false,
        isGeoImaging: false,
        isML: false,
        isOverview: false
    };

    // Event delegation for file buttons, delete buttons, and tab buttons
    document.body.addEventListener('click', function(e) {
        if (e.target.classList.contains('file-btn')) {
            const filename = e.target.dataset.filename;
            document.querySelectorAll('.file-btn').forEach(btn => btn.classList.remove('selected'));
            e.target.classList.add('selected');
            fetchFileInfo(filename);
            updateTabContent();
        } else if (e.target.classList.contains('delete-btn')) {
            const filename = e.target.dataset.filename;
            deleteFile(filename);
        } else if (e.target.classList.contains('tab-btn')) {
            switchTab(e.target.dataset.tab);
        }
    });

    addSourceBtn.addEventListener('click', () => uploadModal.style.display = 'block');
    cancelBtn.addEventListener('click', () => uploadModal.style.display = 'none');

    uploadBtn.addEventListener('click', function() {
        handleFiles(fileInput.files);
    });

    dropZone.addEventListener('dragover', function(e) {
        e.preventDefault();
        dropZone.style.backgroundColor = '#e9e9e9';
    });

    dropZone.addEventListener('dragleave', function(e) {
        e.preventDefault();
        dropZone.style.backgroundColor = '';
    });

    dropZone.addEventListener('drop', function(e) {
        e.preventDefault();
        dropZone.style.backgroundColor = '';
        handleFiles(e.dataTransfer.files);
    });
});

// Function to switch between tabs
function switchTab(tabName) {
    let url;
    // Determine the appropriate URL based on the selected tab
    if (tabName === 'home') {
        url = '/home';
    } else if (tabName === 'overview') {
        const selectedFileBtn = document.querySelector('.file-btn.selected');
        if (selectedFileBtn) {
            const filename = selectedFileBtn.dataset.filename;
            url = `/overview?file=${filename}`;
        } else {
            url = '/overview';
        }
    } else if (tabName === 'eda') {
        const selectedFileBtn = document.querySelector('.file-btn.selected');
        if (selectedFileBtn) {
            const filename = selectedFileBtn.dataset.filename;
            url = `/eda?file/=${filename}`;
        } else {
            url = '/eda';
        }
    } else if (tabName === 'geoimaging') {
        url = '/geoimaging';
    } else if (tabName === 'machinelearning') {
        const selectedFileBtn = document.querySelector('.file-btn.selected');
        if (selectedFileBtn) {
            const filename = selectedFileBtn.dataset.filename;
            url = `/machinelearning?file=${filename}`;
        } else {
            url = '/machinelearning';
        }
    } else {
        url = `/${tabName}`;
    }

    // Fetch the content for the selected tab
    fetch(url)
        .then(response => response.text())
        .then(html => {
            // Debuggging: Print tab name to console
            console.log('Selected Tab:', tabName);
            // Update the tab content with the fetched HTML
            document.getElementById('tab-content').innerHTML = html;
            // Load and execute any scripts in the new content
            loadScripts(document.getElementById('tab-content'));
            // Update the active tab styling
            updateActiveTab(tabName);
        })
        .catch(error => console.error('Error loading tab content:', error));
}

// Function to load and execute scripts from dynamically injected content
function loadScripts(element) {
    // Find all script tags in the injected content
    const scripts = element.getElementsByTagName('script');
    for (let i = 0; i < scripts.length; i++) {
        const script = scripts[i];
        // Create a new script element
        const scriptClone = document.createElement('script');
        // Copy the content of the original script
        scriptClone.text = script.innerHTML;
        // Copy all attributes from the original script
        for (let j = 0; j < script.attributes.length; j++) {
            const attr = script.attributes[j];
            scriptClone.setAttribute(attr.name, attr.value);
        }
        // Replace the original script with the new one to trigger execution
        script.parentNode.replaceChild(scriptClone, script);
    }
}

// Function to update the active tab styling
function updateActiveTab(tabName) {
    // Remove 'active' class from all tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    // Add 'active' class to the selected tab button
    document.querySelector(`.tab-btn[data-tab="${tabName}"]`).classList.add('active');
}

function updateTabContent() {
    const activeTab = document.querySelector('.tab-btn.active');
    if (activeTab) {
        switchTab(activeTab.dataset.tab);
    }
}


function handleFiles(files) {
    if (files.length > 0) {
        Array.from(files).forEach(file => {
            uploadFile(file);
        });
    } else {
        alert('No files selected');
    }
}

function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            addFileToList(data.filename);
            closeUploadModal();
            fetchFileInfo(data.filename);

            // Select the newly uploaded file
            const newFileBtn = document.querySelector(`.file-btn[data-filename="${data.filename}"]`);
            if (newFileBtn) {
                document.querySelectorAll('.file-btn').forEach(btn => btn.classList.remove('selected'));
                newFileBtn.classList.add('selected');
            }
            // Update the tab content
            updateTabContent();
        } else {
            alert('Error uploading file: ' + data.error);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Error uploading file');
    });
}

function addFileToList(filename) {
    const fileItem = document.createElement('div');
    fileItem.className = 'file-item';
    
    // Add different styling for Excel files
    const isExcel = filename.endsWith('.xlsx');
    const buttonClass = isExcel ? 'file-btn excel-file' : 'file-btn';
    
    fileItem.innerHTML = `
        <button class="${buttonClass}" data-filename="${filename}">${filename}</button>
        <span class="download-btn" data-filename="${filename}">
            <i class="fas fa-download"></i>
        </span>
        <span class="delete-btn" data-filename="${filename}">&times;</span>
    `;
    fileList.appendChild(fileItem);
}

// Add click handler for Excel files
document.body.addEventListener('click', function(e) {
    if (e.target.classList.contains('excel-file')) {
        const filename = e.target.dataset.filename;
        showConversionModal(filename);
    } else if (e.target.classList.contains('download-btn') || e.target.parentElement.classList.contains('download-btn')) {
        const filename = e.target.closest('[data-filename]').dataset.filename;
        downloadFile(filename);
    }
});

function closeUploadModal() {
    uploadModal.style.display = 'none';
    fileInput.value = '';
}

function fetchFileInfo(filename) {
    fetch(`/file_info/${filename}`)
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert(data.error);
        } else {
            displayFileInfo(filename, data.file_info);
        }
    })
    .catch(error => console.error('Error fetching file info:', error));
}

function displayFileInfo(filename, fileInfo) {
    const fileInfoContent = document.getElementById('file-info-content');
    const uploadPrompt = document.getElementById('upload-prompt');
    
    if (uploadPrompt) {
        uploadPrompt.remove();
    }

    fileInfoContent.innerHTML = `
        <h3>${filename}</h3>
        <p><strong>Size:</strong> ${fileInfo.size}</p>
        <p><strong>Modified:</strong> ${fileInfo.modified}</p>
        <p><strong>Type:</strong> ${fileInfo.type}</p>
    `;
}    

function deleteFile(filename) {
    fetch(`/delete/${filename}`, {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Find the parent file-item div that contains the clicked delete button
            const fileItem = document.querySelector(`.file-btn[data-filename="${filename}"]`).closest('.file-item');
            if (fileItem) {
                fileItem.remove();
            }
        } else {
            console.error('Server error:', data.error);
        }
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

function showConversionModal(filename) {
    const modal = $('#excelConversionModal');
    modal.find('#convertBtn').off().on('click', function() {
        convertExcelToCSV(filename);
        modal.modal('hide');
    });
    modal.modal('show');
}

function convertExcelToCSV(filename) {
    fetch(`/convert_excel?filename=${filename}`)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Replace Excel file with CSV in sidebar
                const excelBtn = document.querySelector(`.file-btn[data-filename="${filename}"]`);
                excelBtn.className = 'file-btn';
                excelBtn.dataset.filename = data.csvFilename;
                excelBtn.textContent = data.csvFilename;
            }
        });
}

function downloadFile(filename) {
    fetch(`/download_file/${filename}`)
        .then(response => response.blob())
        .then(blob => {
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            a.remove();
        });
}

