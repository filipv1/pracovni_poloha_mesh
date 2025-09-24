# Check status of RunPod job
param(
    [string]$JobId
)

$API_KEY = "YOUR_RUNPOD_API_KEY_HERE"
$ENDPOINT_ID = "dfcn3rqntfybuk"

# If no JobId provided, try to read from file
if (-not $JobId) {
    if (Test-Path "last_job_id.txt") {
        $JobId = Get-Content "last_job_id.txt"
    } else {
        Write-Host "No job ID provided and no last_job_id.txt found" -ForegroundColor Red
        exit 1
    }
}

Write-Host "Checking status for job: $JobId" -ForegroundColor Cyan

$headers = @{
    "Authorization" = "Bearer $API_KEY"
}

# Check status
try {
    $response = Invoke-RestMethod -Uri "https://api.runpod.ai/v2/$ENDPOINT_ID/status/$JobId" -Headers $headers -Method GET

    Write-Host "Status: $($response.status)" -ForegroundColor $(
        switch ($response.status) {
            "IN_QUEUE" { "Yellow" }
            "IN_PROGRESS" { "Cyan" }
            "COMPLETED" { "Green" }
            "FAILED" { "Red" }
            default { "White" }
        }
    )

    if ($response.status -eq "COMPLETED") {
        Write-Host "`nJob completed successfully!" -ForegroundColor Green

        if ($response.output) {
            Write-Host "Output statistics:" -ForegroundColor Cyan
            if ($response.output.statistics) {
                $response.output.statistics | Format-List
            }

            # Save results
            if ($response.output.xlsx_base64) {
                $xlsxBytes = [Convert]::FromBase64String($response.output.xlsx_base64)
                [System.IO.File]::WriteAllBytes("output\result.xlsx", $xlsxBytes)
                Write-Host "Results saved to output\result.xlsx" -ForegroundColor Green
            }

            if ($response.output.pkl_base64) {
                $pklBytes = [Convert]::FromBase64String($response.output.pkl_base64)
                [System.IO.File]::WriteAllBytes("output\result.pkl", $pklBytes)
                Write-Host "PKL saved to output\result.pkl" -ForegroundColor Green
            }
        }
    } elseif ($response.status -eq "FAILED") {
        Write-Host "Job failed!" -ForegroundColor Red
        if ($response.error) {
            Write-Host "Error: $($response.error)" -ForegroundColor Red
        }
    } else {
        Write-Host "Job is still processing. Check again in a few seconds." -ForegroundColor Yellow
    }

} catch {
    Write-Host "Error checking status: $_" -ForegroundColor Red
}