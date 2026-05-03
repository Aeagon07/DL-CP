Write-Host "Starting Docker Infrastructure..." -ForegroundColor Green
docker-compose up -d

Start-Sleep -Seconds 5

Write-Host "Starting Kafka Producer..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'c:\DL CP'; .\venv\Scripts\Activate.ps1; python -m phase1_pipeline.kafka_producer"

Write-Host "Starting API & Dashboard..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'c:\DL CP'; .\venv\Scripts\Activate.ps1; python -m phase5_api.runner"

Write-Host "Boot sequence initiated. Check the new PowerShell windows for progress." -ForegroundColor Cyan
Write-Host "Dashboard will be available at: http://localhost:8000" -ForegroundColor Cyan
