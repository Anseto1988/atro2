# Cleanup Claude temp files and run saturation tests
# Run from project root: powershell -ExecutionPolicy Bypass -File cleanup_and_test.ps1

Write-Host "Cleaning Claude temp directory..."
Remove-Item -Recurse -Force "$env:LOCALAPPDATA\Temp\claude" -ErrorAction SilentlyContinue
Write-Host "Done."

Write-Host "`nRunning F-SelectiveColorSaturation tests..."
Set-Location $PSScriptRoot
python -m pytest tests/unit/processing/test_saturation.py -v --tb=short
