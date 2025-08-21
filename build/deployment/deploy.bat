@echo off
REM AI-Enabled GUI for Medical Image Analysis - Windows Deployment Script
REM This script automates the deployment process for the medical image analysis application

setlocal EnableDelayedExpansion

REM Colors for output (Windows 10+ with ANSI support)
set "RED=[91m"
set "GREEN=[92m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "NC=[0m"

REM Function to print colored output
goto :main

:print_status
echo %BLUE%[INFO]%NC% %~1
goto :eof

:print_success
echo %GREEN%[SUCCESS]%NC% %~1
goto :eof

:print_warning
echo %YELLOW%[WARNING]%NC% %~1
goto :eof

:print_error
echo %RED%[ERROR]%NC% %~1
goto :eof

:command_exists
where %1 >nul 2>&1
goto :eof

:check_requirements
call :print_status "Checking system requirements..."

REM Check Docker
call :command_exists docker
if errorlevel 1 (
    call :print_error "Docker is not installed. Please install Docker Desktop first."
    exit /b 1
)

REM Check Docker Compose
call :command_exists docker-compose
if errorlevel 1 (
    call :print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit /b 1
)

REM Check if Docker is running
docker version >nul 2>&1
if errorlevel 1 (
    call :print_error "Docker is not running. Please start Docker Desktop first."
    exit /b 1
)

call :print_success "System requirements check passed"
goto :eof

:setup_directories
call :print_status "Setting up directory structure..."

REM Create required directories (go up to project root from build/deployment)
cd ..\..
if not exist "data\images\input" mkdir "data\images\input"
if not exist "data\images\annotated" mkdir "data\images\annotated"
if not exist "src\libs\sam_libs\models" mkdir "src\libs\sam_libs\models"
if not exist "backups" mkdir "backups"
if not exist "logs" mkdir "logs"
cd build\deployment

call :print_success "Directory structure created"
goto :eof

:setup_environment
call :print_status "Setting up environment configuration..."

if not exist ".env" (
    if exist ".env.example" (
        copy ".env.example" ".env" >nul
        call :print_success "Environment file created from template"
        call :print_warning "Please review and modify .env file as needed"
    ) else (
        call :print_warning "No .env.example found. Creating basic .env file"
        echo PORT=8000> .env
        echo HOST=0.0.0.0>> .env
        echo DEFAULT_SAM_MODEL=sam2.1_hiera_small>> .env
        echo SESSION_TIMEOUT=28800>> .env
        echo LOG_LEVEL=INFO>> .env
    )
) else (
    call :print_status ".env file already exists"
)
goto :eof

:build_images
call :print_status "Building Docker images..."

docker-compose build
if errorlevel 1 (
    call :print_error "Failed to build Docker images"
    exit /b 1
)

call :print_success "Docker images built successfully"
goto :eof

:start_services
call :print_status "Starting services..."

docker-compose up -d
if errorlevel 1 (
    call :print_error "Failed to start services"
    exit /b 1
)

call :print_success "Services started successfully"
goto :eof

:check_health
call :print_status "Checking service health..."

set /a max_attempts=30
set /a attempt=1

:health_loop
if !attempt! GTR !max_attempts! (
    call :print_error "Service health check failed"
    exit /b 1
)

curl -f http://localhost:8000/aisegmentation >nul 2>&1
if errorlevel 1 (
    call :print_status "Waiting for service to be ready... (attempt !attempt!/!max_attempts!)"
    timeout /t 5 /nobreak >nul
    set /a attempt+=1
    goto :health_loop
)

call :print_success "Service is healthy and ready"
goto :eof

:show_deployment_info
call :print_success "Deployment completed successfully!"
echo.
echo ===================================
echo   Deployment Information
echo ===================================
echo.
echo 🏥 Application URL: http://localhost:8000/aisegmentation
echo.
echo 🔐 Default Login Credentials:
echo    👨‍💼 Administrator: admin/admin123
echo    👩‍⚕️ Radiologist:   doctor/doctor123
echo    👨‍🔬 Researcher:    researcher/research123
echo.
echo 📁 Important Directories:
echo    📥 Input Images:  .\data\images\input\
echo    📤 Results:       .\data\images\annotated\
echo    🔧 Models:        .\sam_libs\models\
echo    📋 Logs:          .\logs\
echo.
echo 🐳 Docker Commands:
echo    View logs:        docker-compose logs -f
echo    Stop services:    docker-compose down
echo    Restart:          docker-compose restart
echo    Update:           docker-compose pull ^&^& docker-compose up -d
echo.
echo ⚠️  Security Reminders:
echo    - Change default passwords after first login
echo    - Review .env configuration for production use
echo    - Ensure proper firewall rules are in place
echo    - Consider using SSL/TLS for production deployment
echo.
goto :eof

:deploy
echo ===================================
echo 🏥 AI-Enabled Medical Image Analysis
echo    Deployment Script
echo ===================================
echo.

call :check_requirements
if errorlevel 1 exit /b 1

call :setup_directories
if errorlevel 1 exit /b 1

call :setup_environment
if errorlevel 1 exit /b 1

call :build_images
if errorlevel 1 exit /b 1

call :start_services
if errorlevel 1 exit /b 1

if not "%SKIP_HEALTH_CHECK%"=="true" (
    call :check_health
    if errorlevel 1 exit /b 1
)

call :show_deployment_info
goto :eof

:start
call :print_status "Starting services..."
docker-compose up -d
if errorlevel 1 (
    call :print_error "Failed to start services"
    exit /b 1
)
call :print_success "Services started"
goto :eof

:stop
call :print_status "Stopping services..."
docker-compose down
if errorlevel 1 (
    call :print_error "Failed to stop services"
    exit /b 1
)
call :print_success "Services stopped"
goto :eof

:restart
call :print_status "Restarting services..."
docker-compose restart
if errorlevel 1 (
    call :print_error "Failed to restart services"
    exit /b 1
)
call :print_success "Services restarted"
goto :eof

:show_logs
call :print_status "Showing application logs..."
docker-compose logs -f
goto :eof

:show_status
call :print_status "Service status:"
docker-compose ps
goto :eof

:cleanup
call :print_warning "This will remove all containers, images, and volumes. Are you sure? (y/N)"
set /p response="Enter your choice: "
if /i "!response!"=="y" (
    call :print_status "Cleaning up..."
    docker-compose down -v --rmi all
    docker system prune -f
    call :print_success "Cleanup completed"
) else (
    call :print_status "Cleanup cancelled"
)
goto :eof

:show_help
echo AI-Enabled Medical Image Analysis - Deployment Script
echo.
echo Usage: %0 [OPTION]
echo.
echo Options:
echo   deploy          Deploy the application (default)
echo   start           Start existing services
echo   stop            Stop running services
echo   restart         Restart services
echo   logs            Show application logs
echo   status          Show service status
echo   cleanup         Remove all containers and images
echo   help            Show this help message
echo.
echo Environment Variables:
echo   SKIP_HEALTH_CHECK    Skip health check during deployment
echo.
echo Examples:
echo   %0 deploy           # Deploy the application
echo   %0 start            # Start services
echo   %0 logs             # View logs
echo   set SKIP_HEALTH_CHECK=true ^&^& %0 deploy  # Deploy without health check
goto :eof

:main
if "%1"=="" goto :deploy
if "%1"=="deploy" goto :deploy
if "%1"=="start" goto :start
if "%1"=="stop" goto :stop
if "%1"=="restart" goto :restart
if "%1"=="logs" goto :show_logs
if "%1"=="status" goto :show_status
if "%1"=="cleanup" goto :cleanup
if "%1"=="help" goto :show_help
if "%1"=="--help" goto :show_help
if "%1"=="-h" goto :show_help

call :print_error "Unknown option: %1"
call :show_help
exit /b 1

:deploy
