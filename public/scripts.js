let videoElement = document.getElementById('webcamVideo');
let canvasElement = document.getElementById('webcamCanvas');
let processedImageElement = document.getElementById('processedImage');
let startButton = document.getElementById('startButton');
let stopButton = document.getElementById('stopButton');
let refreshButton = document.getElementById('refreshButton');
let logElement = document.getElementById('log');
let attendanceBody = document.getElementById('attendanceBody');
let noDataMessage = document.getElementById('noDataMessage');
let canvasContext = canvasElement.getContext('2d');
let ws = null;
let webcamStream = null;
let sendFrameIntervalId = null;
let targetFPS = 15;
let sendFrameRate = 1000 / targetFPS;
let backendUrl = window.location.origin; // Sử dụng URL hiện tại của trang
let wsUrl = backendUrl.replace('http', 'ws') + '/ws/webcam_stream';

function log(message) {
    const now = new Date().toLocaleTimeString();
    logElement.textContent += `[${now}] ${message}\n`;
    logElement.scrollTop = logElement.scrollHeight;
    console.log(`[Log] ${message}`);
}

async function fetchConfig() {
    try {
        const response = await fetch('/api/config', { timeout: 5000 });
        if (!response.ok) throw new Error('Failed to fetch config');
        const config = await response.json();
        backendUrl = config.backendUrl || backendUrl;
        wsUrl = config.wsUrl || wsUrl;
        log(`Config fetched: Backend URL: ${backendUrl}, WS URL: ${wsUrl}`);
    } catch (error) {
        log(`Error fetching config: ${error.message}, using current origin: ${backendUrl}`);
    }
}

async function fetchAttendance() {
    try {
        const response = await fetch(`${backendUrl}/api/attendance`, { timeout: 5000 });
        if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);
        const data = await response.json();
        if (!data || !Array.isArray(data) || data.length === 0) {
            attendanceBody.innerHTML = '';
            noDataMessage.classList.remove('hidden');
            return;
        }
        noDataMessage.classList.add('hidden');
        attendanceBody.innerHTML = data.map(record => `
            <tr class="hover:bg-gray-800 transition-colors">
                <td class="p-2 border-b border-gray-300 text-white">${record.name || 'Unknown'}</td>
                <td class="p-2 border-b border-gray-300 text-white">${record.timestamp || 'N/A'}</td>
            </tr>
        `).join('');
    } catch (error) {
        log(`Error fetching attendance: ${error.message}`);
        noDataMessage.classList.remove('hidden');
        attendanceBody.innerHTML = '';
    }
}

function startStream() {
    log('Starting stream process...');
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
        log('Stream already active or connecting, skipping.');
        return;
    }

    log(`Attempting WebSocket connection to ${wsUrl}...`);
    ws = new WebSocket(wsUrl);
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {
        log('WebSocket connected successfully.');
        startButton.disabled = true;
        stopButton.disabled = false;

        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            log('WebRTC not supported by browser.');
            alert('WebRTC not supported by your browser.');
            return;
        }

        log('Requesting webcam access...');
        navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: targetFPS } }
        }).then(stream => {
            log('Webcam access granted.');
            webcamStream = stream;
            videoElement.srcObject = stream;
            videoElement.onloadedmetadata = () => {
                log(`Webcam resolution: ${videoElement.videoWidth}x${videoElement.videoHeight}`);
                canvasElement.width = videoElement.videoWidth;
                canvasElement.height = videoElement.videoHeight;
                processedImageElement.width = videoElement.videoWidth;
                processedImageElement.height = videoElement.videoHeight;
                sendFrameIntervalId = setInterval(sendFrame, sendFrameRate);
            };
        }).catch(error => {
            log(`Webcam access failed: ${error.name} - ${error.message}`);
            alert(`Webcam access failed: ${error.name} - ${error.message}`);
        });
    };

    ws.onmessage = (event) => {
        log('Received processed frame.');
        const blob = new Blob([event.data], { type: 'image/jpeg' });
        const url = URL.createObjectURL(blob);
        if (processedImageElement.src && processedImageElement.src.startsWith('blob:')) {
            URL.revokeObjectURL(processedImageElement.src);
        }
        processedImageElement.src = url;
    };

    ws.onclose = (event) => {
        log(`WebSocket closed. Code: ${event.code}, Reason: ${event.reason}`);
        stopSendingFrames();
    };

    ws.onerror = (error) => {
        log(`WebSocket error: ${error.message || 'Unknown error'}`);
        console.error('WebSocket error:', error);
        if (ws) ws.close();
    };
}

function sendFrame() {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        log('Cannot send frame: WebSocket not open.');
        return;
    }
    if (!webcamStream || videoElement.videoWidth <= 0 || videoElement.readyState < 2) {
        log('Cannot send frame: Video not ready.');
        return;
    }

    canvasContext.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
    canvasElement.toBlob(blob => {
        if (blob) {
            const reader = new FileReader();
            reader.onload = () => {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(reader.result);
                    log('Frame sent to server.');
                } else {
                    log('Frame not sent: WebSocket not open.');
                }
            };
            reader.onerror = error => log(`Error reading blob: ${error}`);
            reader.readAsArrayBuffer(blob);
        } else {
            log('Failed to create blob from canvas.');
        }
    }, 'image/jpeg', 0.7);
}

function stopSendingFrames() {
    if (sendFrameIntervalId) {
        clearInterval(sendFrameIntervalId);
        sendFrameIntervalId = null;
        log('Stopped sending frames.');
    }
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
        videoElement.srcObject = null;
        log('Webcam stream stopped.');
    }
    if (ws) {
        if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
            ws.close();
        }
        ws = null;
        log('WebSocket client closed.');
    }
    startButton.disabled = false;
    stopButton.disabled = true;
}

// Auto-refresh attendance every 2 seconds
setInterval(fetchAttendance, 2000);

// Initialize and fetch config
fetchConfig().then(() => {
    // Event listeners
    startButton.addEventListener('click', startStream);
    stopButton.addEventListener('click', stopSendingFrames);
    refreshButton.addEventListener('click', fetchAttendance);

    // Initial fetch
    fetchAttendance();
});