# PowerShell script to test RunPod API directly
# No emojis, no diacritics

param(
    [string]$TestType = "health"
)

$API_KEY = "YOUR_RUNPOD_API_KEY_HERE"
$ENDPOINT_ID = "dfcn3rqntfybuk"
$BASE_URL = "https://api.runpod.ai/v2/$ENDPOINT_ID"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "RunPod API Testing" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

function Test-Health {
    Write-Host "Testing endpoint health..." -ForegroundColor Yellow

    $headers = @{
        "Authorization" = "Bearer $API_KEY"
    }

    try {
        $response = Invoke-RestMethod -Uri "$BASE_URL/health" -Headers $headers -Method GET
        Write-Host "[OK] Endpoint is healthy" -ForegroundColor Green
        Write-Host "Workers ready: $($response.workers.ready)" -ForegroundColor Cyan
        Write-Host "Workers running: $($response.workers.running)" -ForegroundColor Cyan
        return $true
    }
    catch {
        Write-Host "[ERROR] Health check failed: $_" -ForegroundColor Red
        return $false
    }
}

function Test-MinimalJob {
    Write-Host "Submitting minimal test job..." -ForegroundColor Yellow

    $headers = @{
        "Authorization" = "Bearer $API_KEY"
        "Content-Type" = "application/json"
    }

    # Create minimal test data
    $testVideo = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes("test video data"))

    $body = @{
        input = @{
            video_base64 = $testVideo
            video_name = "test.mp4"
            quality = "medium"
            user_email = "test@example.com"
        }
    } | ConvertTo-Json -Depth 10

    try {
        $response = Invoke-RestMethod -Uri "$BASE_URL/run" -Headers $headers -Method POST -Body $body
        Write-Host "[OK] Job submitted successfully" -ForegroundColor Green
        Write-Host "Job ID: $($response.id)" -ForegroundColor Cyan
        Write-Host "Status: $($response.status)" -ForegroundColor Cyan

        # Save job ID
        $response.id | Out-File "last_test_job.txt"

        return $response.id
    }
    catch {
        Write-Host "[ERROR] Job submission failed: $_" -ForegroundColor Red
        return $null
    }
}

function Check-JobStatus {
    param([string]$JobId)

    if (-not $JobId) {
        if (Test-Path "last_test_job.txt") {
            $JobId = Get-Content "last_test_job.txt"
        }
        else {
            Write-Host "[ERROR] No job ID provided" -ForegroundColor Red
            return
        }
    }

    Write-Host "Checking job status: $JobId" -ForegroundColor Yellow

    $headers = @{
        "Authorization" = "Bearer $API_KEY"
    }

    $maxAttempts = 30
    $attempt = 0

    while ($attempt -lt $maxAttempts) {
        try {
            $response = Invoke-RestMethod -Uri "$BASE_URL/status/$JobId" -Headers $headers -Method GET

            Write-Host "Status: $($response.status)" -ForegroundColor Cyan

            if ($response.status -eq "COMPLETED") {
                Write-Host "[OK] Job completed successfully!" -ForegroundColor Green

                if ($response.output) {
                    Write-Host "Output available:" -ForegroundColor Green
                    if ($response.output.xlsx_base64) {
                        Write-Host "  - XLSX data present" -ForegroundColor Cyan
                    }
                    if ($response.output.pkl_base64) {
                        Write-Host "  - PKL data present" -ForegroundColor Cyan
                    }
                    if ($response.output.statistics) {
                        Write-Host "  - Statistics: $($response.output.statistics | ConvertTo-Json -Compress)" -ForegroundColor Cyan
                    }
                }
                return $true
            }
            elseif ($response.status -eq "FAILED") {
                Write-Host "[ERROR] Job failed!" -ForegroundColor Red
                if ($response.error) {
                    Write-Host "Error: $($response.error)" -ForegroundColor Red
                }
                return $false
            }
            else {
                Write-Host "Job still processing... (attempt $attempt/$maxAttempts)" -ForegroundColor Yellow
                Start-Sleep -Seconds 2
                $attempt++
            }
        }
        catch {
            Write-Host "[ERROR] Status check failed: $_" -ForegroundColor Red
            return $false
        }
    }

    Write-Host "[TIMEOUT] Job did not complete in time" -ForegroundColor Yellow
    return $false
}

function Show-Logs {
    Write-Host "To view RunPod logs:" -ForegroundColor Yellow
    Write-Host "1. Go to https://www.runpod.io/console/serverless" -ForegroundColor Cyan
    Write-Host "2. Click on your endpoint" -ForegroundColor Cyan
    Write-Host "3. Click on 'Logs' tab" -ForegroundColor Cyan
    Write-Host ""
}

# Main execution
Write-Host "Test type: $TestType" -ForegroundColor Cyan
Write-Host ""

switch ($TestType) {
    "health" {
        Test-Health
    }
    "submit" {
        $jobId = Test-MinimalJob
        if ($jobId) {
            Write-Host ""
            Write-Host "Waiting 5 seconds before checking status..." -ForegroundColor Yellow
            Start-Sleep -Seconds 5
            Check-JobStatus -JobId $jobId
        }
    }
    "status" {
        Check-JobStatus
    }
    "full" {
        Write-Host "Running full test suite..." -ForegroundColor Cyan
        Write-Host ""

        $healthOk = Test-Health
        if ($healthOk) {
            Write-Host ""
            $jobId = Test-MinimalJob
            if ($jobId) {
                Write-Host ""
                Write-Host "Waiting 5 seconds..." -ForegroundColor Yellow
                Start-Sleep -Seconds 5
                Check-JobStatus -JobId $jobId
            }
        }
    }
    default {
        Write-Host "Usage: .\test_runpod_api.ps1 -TestType [health|submit|status|full]" -ForegroundColor Yellow
    }
}

Write-Host ""
Show-Logs