/**
 * Progress Tracker using Server-Sent Events
 */

class ProgressTracker {
    constructor(jobId) {
        this.jobId = jobId;
        this.eventSource = null;
        this.stages = ['uploading', 'mediapipe', 'smplx', 'angles', 'analysis', 'downloading'];
        this.currentStage = null;
        
        // Start tracking
        this.startTracking();
    }
    
    startTracking() {
        // Connect to SSE endpoint
        this.eventSource = new EventSource(`/api/progress/${this.jobId}`);
        
        // Handle messages
        this.eventSource.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.handleUpdate(data);
            } catch (err) {
                console.error('Error parsing SSE data:', err);
            }
        };
        
        // Handle errors
        this.eventSource.onerror = (error) => {
            console.error('SSE Error:', error);
            if (this.eventSource.readyState === EventSource.CLOSED) {
                this.stopTracking();
            }
        };
        
        // Handle connection open
        this.eventSource.onopen = () => {
            console.log('SSE connection established');
        };
    }
    
    stopTracking() {
        if (this.eventSource) {
            this.eventSource.close();
            this.eventSource = null;
        }
    }
    
    handleUpdate(data) {
        switch (data.type) {
            case 'connected':
                console.log('Connected to progress stream');
                break;
                
            case 'progress':
                this.updateProgress(data);
                break;
                
            case 'completed':
                this.handleCompletion(data);
                break;
                
            case 'error':
                this.handleError(data);
                break;
                
            case 'heartbeat':
                // Keep connection alive
                break;
                
            default:
                console.log('Unknown message type:', data.type);
        }
    }
    
    updateProgress(data) {
        // Update status icon and text
        this.updateStatus(data.status);
        
        // Update stage message
        const stageMessage = document.getElementById('stage-message');
        if (stageMessage && data.message) {
            stageMessage.textContent = data.message;
        }
        
        // Update progress bar
        const progressBar = document.getElementById('progress-bar');
        const progressPercent = document.getElementById('progress-percent');
        if (progressBar && progressPercent) {
            const percent = Math.round(data.percent || 0);
            progressBar.style.width = `${percent}%`;
            progressPercent.textContent = `${percent}%`;
        }
        
        // Update stage indicators
        if (data.stage && data.stage !== this.currentStage) {
            this.updateStageIndicators(data.stage);
            this.currentStage = data.stage;
        }
        
        // Update time estimates
        this.updateTimeEstimates(data);
        
        // Handle frames progress
        if (data.frames_processed !== undefined && data.total_frames) {
            const framesText = `${data.frames_processed}/${data.total_frames} frames`;
            const stageMessage = document.getElementById('stage-message');
            if (stageMessage && data.stage === 'mediapipe') {
                stageMessage.textContent = `Processing frames: ${framesText}`;
            }
        }
    }
    
    updateStatus(status) {
        const statusIcon = document.getElementById('status-icon');
        const statusText = document.getElementById('status-text');
        
        if (statusIcon) {
            switch (status) {
                case 'queued':
                    statusIcon.innerHTML = '<i class="fas fa-hourglass-half text-yellow-500 animate-pulse-slow"></i>';
                    break;
                case 'processing':
                    statusIcon.innerHTML = '<i class="fas fa-cog fa-spin text-indigo-600"></i>';
                    break;
                case 'completed':
                    statusIcon.innerHTML = '<i class="fas fa-check-circle text-green-500"></i>';
                    break;
                case 'failed':
                    statusIcon.innerHTML = '<i class="fas fa-times-circle text-red-500"></i>';
                    break;
            }
        }
        
        if (statusText) {
            switch (status) {
                case 'queued':
                    statusText.textContent = 'Waiting in Queue';
                    break;
                case 'processing':
                    statusText.textContent = 'Processing';
                    break;
                case 'completed':
                    statusText.textContent = 'Completed Successfully';
                    break;
                case 'failed':
                    statusText.textContent = 'Processing Failed';
                    break;
            }
        }
    }
    
    updateStageIndicators(currentStage) {
        const stageItems = document.querySelectorAll('.stage-item');
        
        stageItems.forEach(item => {
            const stage = item.getAttribute('data-stage');
            const icon = item.querySelector('.stage-icon');
            
            if (!stage || !icon) return;
            
            const stageIndex = this.stages.indexOf(stage);
            const currentIndex = this.stages.indexOf(currentStage);
            
            if (stageIndex < currentIndex) {
                // Completed stage
                icon.classList.remove('bg-gray-200', 'bg-indigo-600', 'text-white');
                icon.classList.add('bg-green-500', 'text-white');
                icon.innerHTML = '<i class="fas fa-check text-sm"></i>';
            } else if (stageIndex === currentIndex) {
                // Current stage
                icon.classList.remove('bg-gray-200', 'bg-green-500');
                icon.classList.add('bg-indigo-600', 'text-white');
                const originalIcon = this.getStageIcon(stage);
                if (!icon.querySelector('.fa-spin')) {
                    icon.innerHTML = originalIcon.replace('text-sm', 'text-sm fa-spin');
                }
            } else {
                // Pending stage
                icon.classList.remove('bg-green-500', 'bg-indigo-600', 'text-white');
                icon.classList.add('bg-gray-200');
                icon.innerHTML = this.getStageIcon(stage);
            }
        });
    }
    
    getStageIcon(stage) {
        const icons = {
            'uploading': '<i class="fas fa-upload text-sm"></i>',
            'mediapipe': '<i class="fas fa-eye text-sm"></i>',
            'smplx': '<i class="fas fa-user text-sm"></i>',
            'angles': '<i class="fas fa-ruler-combined text-sm"></i>',
            'analysis': '<i class="fas fa-chart-bar text-sm"></i>',
            'downloading': '<i class="fas fa-download text-sm"></i>'
        };
        return icons[stage] || '<i class="fas fa-circle text-sm"></i>';
    }
    
    updateTimeEstimates(data) {
        const timeElapsed = document.getElementById('time-elapsed');
        const timeRemaining = document.getElementById('time-remaining');
        
        if (timeElapsed && data.time_elapsed !== undefined) {
            timeElapsed.textContent = this.formatTime(data.time_elapsed);
        }
        
        if (timeRemaining && data.time_remaining !== undefined) {
            timeRemaining.textContent = this.formatTime(data.time_remaining);
        }
    }
    
    formatTime(seconds) {
        if (seconds === null || seconds === undefined) {
            return '--:--';
        }
        
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);
        
        if (hours > 0) {
            return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
        } else {
            return `${minutes}:${secs.toString().padStart(2, '0')}`;
        }
    }
    
    handleCompletion(data) {
        this.stopTracking();
        
        // Update UI to show completion
        this.updateStatus('completed');
        this.updateStageIndicators('completed');
        
        // Update progress to 100%
        const progressBar = document.getElementById('progress-bar');
        const progressPercent = document.getElementById('progress-percent');
        if (progressBar && progressPercent) {
            progressBar.style.width = '100%';
            progressPercent.textContent = '100%';
        }
        
        // Show download section
        if (data.files && data.files.length > 0) {
            this.showDownloads(data.files);
        }
    }
    
    handleError(data) {
        this.stopTracking();
        
        // Update UI to show error
        this.updateStatus('failed');
        
        // Show error section
        const errorSection = document.getElementById('error-section');
        const errorMessage = document.getElementById('error-message');
        
        if (errorSection && errorMessage) {
            errorMessage.textContent = data.message || 'An unknown error occurred';
            errorSection.classList.remove('hidden');
        }
    }
    
    showDownloads(files) {
        const downloadSection = document.getElementById('download-section');
        const downloadLinks = document.getElementById('download-links');
        
        if (!downloadSection || !downloadLinks) return;
        
        // Clear existing links
        downloadLinks.innerHTML = '';
        
        // Add download links
        files.forEach(file => {
            const icon = file.type === 'xlsx' ? 'fa-file-excel' : 'fa-file-archive';
            const color = file.type === 'xlsx' ? 'text-green-600' : 'text-blue-600';
            const bgColor = file.type === 'xlsx' ? 'bg-green-50' : 'bg-blue-50';
            
            const linkDiv = document.createElement('div');
            linkDiv.className = `flex items-center justify-between p-4 ${bgColor} rounded-lg hover:shadow-md transition-shadow`;
            linkDiv.innerHTML = `
                <div class="flex items-center">
                    <i class="fas ${icon} ${color} text-2xl mr-4"></i>
                    <div>
                        <p class="font-medium text-gray-900">${file.filename}</p>
                        <p class="text-sm text-gray-500">${file.size_mb} MB</p>
                    </div>
                </div>
                <a href="${file.url}" 
                   class="px-4 py-2 bg-white border border-gray-300 rounded-md text-sm font-medium text-gray-700 hover:bg-gray-50 flex items-center">
                    <i class="fas fa-download mr-2"></i>Download
                </a>
            `;
            downloadLinks.appendChild(linkDiv);
        });
        
        // Show the section
        downloadSection.classList.remove('hidden');
        
        // Scroll to downloads
        downloadSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
}

// Auto-initialize if job ID is present
document.addEventListener('DOMContentLoaded', () => {
    // The job ID will be set in the template
    if (typeof jobId !== 'undefined') {
        window.progressTracker = new ProgressTracker(jobId);
    }
});