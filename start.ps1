# Vibe-Sync Startup Script
# Starts the full-stack application

Write-Host "=" -ForegroundColor Green -NoNewline; Write-Host ("=" * 59) -ForegroundColor Green
Write-Host "🎵 VIBE-SYNC - STARTING REAL PRODUCTION SYSTEM" -ForegroundColor Cyan
Write-Host "=" -ForegroundColor Green -NoNewline; Write-Host ("=" * 59) -ForegroundColor Green
Write-Host ""

# Check if .env exists
if (-not (Test-Path ".env")) {
    Write-Host "⚠️  WARNING: .env file not found!" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "📝 You need to create a .env file with your Spotify credentials:" -ForegroundColor White
    Write-Host ""
    Write-Host "1. Go to: https://developer.spotify.com/dashboard" -ForegroundColor Cyan
    Write-Host "2. Create an app and get your Client ID & Secret" -ForegroundColor Cyan
    Write-Host "3. Create .env file with:" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "   SPOTIFY_CLIENT_ID=your_client_id" -ForegroundColor Gray
    Write-Host "   SPOTIFY_CLIENT_SECRET=your_client_secret" -ForegroundColor Gray
    Write-Host ""
    
    $continue = Read-Host "Continue anyway? (y/n)"
    if ($continue -ne "y") {
        exit
    }
}

Write-Host "✅ Starting backend server..." -ForegroundColor Green
Write-Host ""
Write-Host "📡 Server will be available at: http://localhost:8000" -ForegroundColor Cyan
Write-Host "🌐 Open this URL in your browser!" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""
Write-Host "=" -ForegroundColor Green -NoNewline; Write-Host ("=" * 59) -ForegroundColor Green
Write-Host ""

# Start the server
.\.venv\Scripts\python.exe backend/server.py
