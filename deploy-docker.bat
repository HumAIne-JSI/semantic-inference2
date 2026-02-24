@echo off
REM Quick Docker deployment script for Windows

echo ========================================
echo  RAG SEMANTIC INFERENCE - DOCKER SETUP
echo ========================================
echo.

REM Check Docker is installed
docker --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker is not installed or not in PATH
    echo Please install Docker Desktop from https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)

REM Check Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker is not running
    echo Please start Docker Desktop
    pause
    exit /b 1
)

echo [1/5] Docker is running

REM Check if .env exists
if not exist .env (
    echo.
    echo WARNING: .env file not found
    echo Creating .env from .env.example...
    if exist .env.example (
        copy .env.example .env >nul
        echo.
        echo IMPORTANT: Edit .env file and update:
        echo   - NEO4J_PASSWORD
        echo   - GEMINI_API_KEY
        echo   - OPENAI_API_KEY
        echo.
        echo Press any key to open .env in notepad...
        pause >nul
        notepad .env
    ) else (
        echo ERROR: .env.example not found
        pause
        exit /b 1
    )
)

echo [2/5] Environment configured

REM Pull latest images
echo.
echo [3/5] Pulling Docker images...
docker-compose pull

REM Build custom images
echo.
echo [4/5] Building application images...
docker-compose build

REM Start services
echo.
echo [5/5] Starting services...
docker-compose up -d

echo.
echo ========================================
echo  DEPLOYMENT COMPLETE
echo ========================================
echo.
echo Services status:
docker-compose ps
echo.
echo Access points:
echo   Neo4j Browser: http://localhost:7474
echo   RAG API:       http://localhost:5001
echo   API Health:    http://localhost:5001/health
echo.
echo View logs:
echo   docker-compose logs -f
echo.
echo Stop services:
echo   docker-compose down
echo.
echo ========================================
echo.
echo Waiting for services to be healthy...
timeout /t 10 /nobreak >nul

REM Check health
echo.
echo Checking service health...
curl -s http://localhost:5001 >nul 2>&1
if errorlevel 1 (
    echo WARNING: API may not be ready yet
    echo Check logs: docker-compose logs rag-api
) else (
    echo SUCCESS: API is responding
)

echo.
echo Press any key to view live logs (Ctrl+C to exit)...
pause >nul
docker-compose logs -f
