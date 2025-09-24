# Test RunPod endpoint with real video
param(
    [string]$VideoPath = "test\sample_video.mp4"
)

$API_KEY = "YOUR_RUNPOD_API_KEY_HERE"
$ENDPOINT_ID = "dfcn3rqntfybuk"

# Check if video exists
if (-not (Test-Path $VideoPath)) {
    Write-Host "Error: Video file not found at $VideoPath" -ForegroundColor Red
    Write-Host "Please provide a valid video path as parameter" -ForegroundColor Yellow
    Write-Host "Usage: .\test_runpod.ps1 -VideoPath 'C:\path\to\video.mp4'" -ForegroundColor Yellow
    exit 1
}

Write-Host "Testing RunPod endpoint with video: $VideoPath" -ForegroundColor Green

# Read and encode video
$videoBytes = [System.IO.File]::ReadAllBytes($VideoPath)
$videoBase64 = [Convert]::ToBase64String($videoBytes)
$videoName = Split-Path $VideoPath -Leaf

Write-Host "Video encoded, size: $([math]::Round($videoBytes.Length / 1MB, 2)) MB" -ForegroundColor Cyan

# Prepare request
$headers = @{
    "Authorization" = "Bearer $API_KEY"
    "Content-Type" = "application/json"
}

$body = @{
    input = @{
        video_base64 = $videoBase64
        video_name = $videoName
        quality = "medium"
        user_email = "test@example.com"
    }
} | ConvertTo-Json -Depth 10

Write-Host "Sending request to RunPod..." -ForegroundColor Yellow

# Send request
try {
    $response = Invoke-RestMethod -Uri "https://api.runpod.ai/v2/$ENDPOINT_ID/run" -Headers $headers -Method POST -Body $body

    Write-Host "Success! Job submitted" -ForegroundColor Green
    Write-Host "Job ID: $($response.id)" -ForegroundColor Cyan
    Write-Host "Status: $($response.status)" -ForegroundColor Cyan

    # Save job ID for checking status
    $response.id | Out-File "last_job_id.txt"

    Write-Host "`nTo check status, run:" -ForegroundColor Yellow
    Write-Host ".\check_status.ps1" -ForegroundColor White

} catch {
    Write-Host "Error: $_" -ForegroundColor Red
}