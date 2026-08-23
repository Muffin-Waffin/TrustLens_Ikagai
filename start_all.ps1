# Trustlens - Start Both Frontend and Backend
# Run this script from the prototype directory

Write-Host "Starting Trustlens Full Stack..." -ForegroundColor Green

# Check if we're in the right directory
$prototypeDir = $PSScriptRoot
Set-Location $prototypeDir

# Function to check if a port is in use
function Test-Port {
    param($Port)
    $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::IPv6Loopback, $Port)
    try {
        $listener.Start()
        $listener.Stop()
        return $false
    } catch {
        return $true
    }
}

# Kill any existing processes on ports 8000 and 5173
Write-Host "Checking for existing processes..." -ForegroundColor Yellow
$ports = @(8000, 5173)
foreach ($port in $ports) {
    if (Test-Port $port) {
        Write-Host "Port $port is in use. Killing process..." -ForegroundColor Yellow
        $process = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty OwningProcess
        if ($process) {
            Stop-Process -Id $process -Force -ErrorAction SilentlyContinue
            Start-Sleep 1
        }
    }
}

# Start Backend
Write-Host "`nStarting Backend (FastAPI on port 8000)..." -ForegroundColor Cyan
$backendDir = "$prototypeDir\synthguard_backend"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$backendDir'; pip install -r requirements.txt -q; uvicorn app.main:app --reload --port 8000" -WorkingDirectory $backendDir

# Wait for backend to start
Write-Host "Waiting for backend to start..." -ForegroundColor Yellow
$backendReady = $false
for ($i = 0; $i -lt 30; $i++) {
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:8000/api/health" -TimeoutSec 2 -ErrorAction Stop
        if ($response.status -eq "ok") {
            Write-Host "Backend is ready! Model loaded: $($response.model_loaded), Device: $($response.device)" -ForegroundColor Green
            $backendReady = $true
            break
        }
    } catch {
        Start-Sleep 1
    }
}

if (-not $backendReady) {
    Write-Host "Backend failed to start. Check the backend window for errors." -ForegroundColor Red
    exit 1
}

# Start Frontend
Write-Host "`nStarting Frontend (Vite on port 5173)..." -ForegroundColor Cyan
$frontendDir = "$prototypeDir\frontend"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$frontendDir'; npm install; npm run dev" -WorkingDirectory $frontendDir

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "Trustlens is running!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "Frontend: http://localhost:5173" -ForegroundColor Cyan
Write-Host "Backend API: http://localhost:8000" -ForegroundColor Cyan
Write-Host "API Docs: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Green
Write-Host "Check the two PowerShell windows for logs." -ForegroundColor Yellow
Write-Host "Press Ctrl+C in each window to stop." -ForegroundColor Yellow