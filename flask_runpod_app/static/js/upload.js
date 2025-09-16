/**
 * Video Upload Handler
 */

document.addEventListener('DOMContentLoaded', function() {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const dropZoneContent = document.getElementById('drop-zone-content');
    const filePreview = document.getElementById('file-preview');
    const uploadProgress = document.getElementById('upload-progress');
    const uploadBtn = document.getElementById('upload-btn');
    const cancelBtn = document.getElementById('cancel-btn');
    const fileName = document.getElementById('file-name');
    const fileSize = document.getElementById('file-size');
    const progressBar = document.getElementById('progress-bar');
    const progressText = document.getElementById('progress-text');
    
    let selectedFile = null;
    let uploadXHR = null;
    
    // Configure maximum file size (5GB)
    const MAX_FILE_SIZE = 5 * 1024 * 1024 * 1024; // 5GB in bytes
    
    // Drag and drop handlers
    dropZone.addEventListener('click', () => {
        if (!selectedFile) {
            fileInput.click();
        }
    });
    
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('border-indigo-500', 'bg-indigo-50');
    });
    
    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropZone.classList.remove('border-indigo-500', 'bg-indigo-50');
    });
    
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('border-indigo-500', 'bg-indigo-50');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    });
    
    // File input change handler
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });
    
    // Upload button handler
    uploadBtn.addEventListener('click', () => {
        if (selectedFile) {
            uploadFile();
        }
    });
    
    // Cancel button handler
    cancelBtn.addEventListener('click', () => {
        resetUpload();
    });
    
    /**
     * Handle file selection
     */
    function handleFileSelect(file) {
        // Validate file type
        if (!file.name.toLowerCase().endsWith('.mp4')) {
            showError('Please select an MP4 video file');
            return;
        }
        
        // Validate file size
        if (file.size > MAX_FILE_SIZE) {
            showError(`File too large. Maximum size is ${formatFileSize(MAX_FILE_SIZE)}`);
            return;
        }
        
        selectedFile = file;
        showFilePreview();
    }
    
    /**
     * Show file preview
     */
    function showFilePreview() {
        fileName.textContent = selectedFile.name;
        fileSize.textContent = formatFileSize(selectedFile.size);
        
        dropZoneContent.classList.add('hidden');
        filePreview.classList.remove('hidden');
        uploadProgress.classList.add('hidden');
    }
    
    /**
     * Upload file to server
     */
    function uploadFile() {
        const formData = new FormData();
        formData.append('video', selectedFile);
        
        // Hide preview, show progress
        filePreview.classList.add('hidden');
        uploadProgress.classList.remove('hidden');
        
        // Create XMLHttpRequest
        uploadXHR = new XMLHttpRequest();
        
        // Upload progress handler
        uploadXHR.upload.addEventListener('progress', (e) => {
            if (e.lengthComputable) {
                const percentComplete = Math.round((e.loaded / e.total) * 100);
                updateProgress(percentComplete);
            }
        });
        
        // Upload complete handler
        uploadXHR.addEventListener('load', () => {
            if (uploadXHR.status === 200) {
                try {
                    const response = JSON.parse(uploadXHR.responseText);
                    if (response.success && response.redirect) {
                        // Redirect to progress page
                        window.location.href = response.redirect;
                    } else {
                        showError(response.error || 'Upload failed');
                        resetUpload();
                    }
                } catch (err) {
                    showError('Invalid server response');
                    resetUpload();
                }
            } else {
                showError('Upload failed with status: ' + uploadXHR.status);
                resetUpload();
            }
        });
        
        // Error handler
        uploadXHR.addEventListener('error', () => {
            showError('Upload failed. Please try again.');
            resetUpload();
        });
        
        // Abort handler
        uploadXHR.addEventListener('abort', () => {
            showInfo('Upload cancelled');
            resetUpload();
        });
        
        // Send request
        uploadXHR.open('POST', '/api/upload', true);
        uploadXHR.send(formData);
    }
    
    /**
     * Update progress bar
     */
    function updateProgress(percent) {
        progressBar.style.width = percent + '%';
        progressText.textContent = percent + '%';
    }
    
    /**
     * Reset upload interface
     */
    function resetUpload() {
        selectedFile = null;
        fileInput.value = '';
        
        dropZoneContent.classList.remove('hidden');
        filePreview.classList.add('hidden');
        uploadProgress.classList.add('hidden');
        
        updateProgress(0);
        
        if (uploadXHR) {
            uploadXHR.abort();
            uploadXHR = null;
        }
    }
    
    /**
     * Format file size for display
     */
    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    
    /**
     * Show error message
     */
    function showError(message) {
        const alertDiv = document.createElement('div');
        alertDiv.className = 'fixed top-4 right-4 bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded z-50';
        alertDiv.innerHTML = `
            <div class="flex items-center">
                <i class="fas fa-exclamation-circle mr-2"></i>
                <span>${message}</span>
                <button onclick="this.parentElement.parentElement.remove()" class="ml-4">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `;
        document.body.appendChild(alertDiv);
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentElement) {
                alertDiv.remove();
            }
        }, 5000);
    }
    
    /**
     * Show info message
     */
    function showInfo(message) {
        const alertDiv = document.createElement('div');
        alertDiv.className = 'fixed top-4 right-4 bg-blue-100 border border-blue-400 text-blue-700 px-4 py-3 rounded z-50';
        alertDiv.innerHTML = `
            <div class="flex items-center">
                <i class="fas fa-info-circle mr-2"></i>
                <span>${message}</span>
                <button onclick="this.parentElement.parentElement.remove()" class="ml-4">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `;
        document.body.appendChild(alertDiv);
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentElement) {
                alertDiv.remove();
            }
        }, 5000);
    }
});