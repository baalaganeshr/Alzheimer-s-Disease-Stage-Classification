# Docker Run Script for Alzheimer's Disease Classification
# This script builds and runs the Docker container

$separator = "=" * 80
Write-Host $separator -ForegroundColor Cyan
Write-Host "Alzheimer's Disease Classification - Docker Setup" -ForegroundColor Cyan
Write-Host $separator -ForegroundColor Cyan

# Check if Docker is running
Write-Host "`nChecking Docker..." -ForegroundColor Yellow
try {
    docker info | Out-Null
    Write-Host "* Docker is running" -ForegroundColor Green
}
catch {
    Write-Host "X Docker is not running. Please start Docker Desktop." -ForegroundColor Red
    exit 1
}

# Check if dataset exists
$datasetPath = "..\dataset\Data"
if (Test-Path $datasetPath) {
    $fileCount = (Get-ChildItem $datasetPath -Recurse -File).Count
    Write-Host "* Dataset found: $fileCount files" -ForegroundColor Green
}
else {
    Write-Host "X Dataset not found at: $datasetPath" -ForegroundColor Red
    Write-Host "  Please ensure dataset is extracted to: ..\dataset\Data\" -ForegroundColor Yellow
    exit 1
}

# Build Docker image
Write-Host "`nBuilding Docker image..." -ForegroundColor Yellow
docker-compose build

if ($LASTEXITCODE -eq 0) {
    Write-Host "* Docker image built successfully" -ForegroundColor Green
}
else {
    Write-Host "X Failed to build Docker image" -ForegroundColor Red
    exit 1
}

# Run the container
Write-Host "`nStarting training in Docker container..." -ForegroundColor Yellow
Write-Host "This may take a while. Results will be saved to ./outputs/" -ForegroundColor Cyan
Write-Host ""

docker-compose up

$endSeparator = "`n" + $separator
Write-Host $endSeparator -ForegroundColor Cyan
Write-Host "Training completed!" -ForegroundColor Green
Write-Host "Check ./outputs/ for results and visualizations" -ForegroundColor Cyan
Write-Host "Check ./models/ for saved models" -ForegroundColor Cyan
Write-Host $separator -ForegroundColor Cyan
