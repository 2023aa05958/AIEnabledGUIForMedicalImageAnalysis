#!/bin/bash

# AI-Enabled GUI for Medical Image Analysis - Deployment Script
# This script automates the deployment process for the medical image analysis application

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check system requirements
check_requirements() {
    print_status "Checking system requirements..."
    
    # Check Docker
    if ! command_exists docker; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command_exists docker-compose; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check Git
    if ! command_exists git; then
        print_warning "Git is not installed. Some features may not work properly."
    fi
    
    print_success "System requirements check passed"
}

# Function to setup directories
setup_directories() {
    print_status "Setting up directory structure..."
    
    # Create required directories (go up to project root from build/deployment)
    cd ../..
    mkdir -p data/images/input
    mkdir -p data/images/annotated
    mkdir -p src/libs/sam_libs/models
    mkdir -p backups
    mkdir -p logs
    cd build/deployment
    
    # Set permissions
    chmod 755 ../../data/images/input
    chmod 755 ../../data/images/annotated
    chmod 755 ../../src/libs/sam_libs/models
    chmod 755 ../../backups
    chmod 755 ../../logs
    
    print_success "Directory structure created"
}

# Function to setup environment
setup_environment() {
    print_status "Setting up environment configuration..."
    
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            cp .env.example .env
            print_success "Environment file created from template"
            print_warning "Please review and modify .env file as needed"
        else
            print_warning "No .env.example found. Creating basic .env file"
            cat > .env << EOF
PORT=8000
HOST=0.0.0.0
DEFAULT_SAM_MODEL=sam2.1_hiera_small
SESSION_TIMEOUT=28800
LOG_LEVEL=INFO
EOF
        fi
    else
        print_status ".env file already exists"
    fi
}

# Function to build Docker images
build_images() {
    print_status "Building Docker images..."
    
    # Build the main application image
    docker-compose build
    
    print_success "Docker images built successfully"
}

# Function to start services
start_services() {
    print_status "Starting services..."
    
    # Start the application
    docker-compose up -d
    
    print_success "Services started successfully"
}

# Function to check service health
check_health() {
    print_status "Checking service health..."
    
    # Wait for service to be ready
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f http://localhost:8000/aisegmentation >/dev/null 2>&1; then
            print_success "Service is healthy and ready"
            return 0
        fi
        
        print_status "Waiting for service to be ready... (attempt $attempt/$max_attempts)"
        sleep 5
        attempt=$((attempt + 1))
    done
    
    print_error "Service health check failed"
    return 1
}

# Function to display deployment information
show_deployment_info() {
    print_success "Deployment completed successfully!"
    echo
    echo "==================================="
    echo "  Deployment Information"
    echo "==================================="
    echo
    echo "🏥 Application URL: http://localhost:8000/aisegmentation"
    echo
    echo "🔐 Default Login Credentials:"
    echo "   👨‍💼 Administrator: admin/admin123"
    echo "   👩‍⚕️ Radiologist:   doctor/doctor123"
    echo "   👨‍🔬 Researcher:    researcher/research123"
    echo
    echo "📁 Important Directories:"
    echo "   📥 Input Images:  ./data/images/input/"
    echo "   📤 Results:       ./data/images/annotated/"
    echo "   🔧 Models:        ./sam_libs/models/"
    echo "   📋 Logs:          ./logs/"
    echo
    echo "🐳 Docker Commands:"
    echo "   View logs:        docker-compose logs -f"
    echo "   Stop services:    docker-compose down"
    echo "   Restart:          docker-compose restart"
    echo "   Update:           docker-compose pull && docker-compose up -d"
    echo
    echo "⚠️  Security Reminders:"
    echo "   - Change default passwords after first login"
    echo "   - Review .env configuration for production use"
    echo "   - Ensure proper firewall rules are in place"
    echo "   - Consider using SSL/TLS for production deployment"
    echo
}

# Function to cleanup on error
cleanup_on_error() {
    print_error "Deployment failed. Cleaning up..."
    docker-compose down >/dev/null 2>&1 || true
    exit 1
}

# Main deployment function
deploy() {
    echo "==================================="
    echo "🏥 AI-Enabled Medical Image Analysis"
    echo "   Deployment Script"
    echo "==================================="
    echo
    
    # Set trap for cleanup on error
    trap cleanup_on_error ERR
    
    # Run deployment steps
    check_requirements
    setup_directories
    setup_environment
    build_images
    start_services
    
    # Check if health check should be performed
    if [ "${SKIP_HEALTH_CHECK:-false}" != "true" ]; then
        check_health
    fi
    
    show_deployment_info
}

# Function to show help
show_help() {
    echo "AI-Enabled Medical Image Analysis - Deployment Script"
    echo
    echo "Usage: $0 [OPTION]"
    echo
    echo "Options:"
    echo "  deploy          Deploy the application (default)"
    echo "  start           Start existing services"
    echo "  stop            Stop running services"
    echo "  restart         Restart services"
    echo "  logs            Show application logs"
    echo "  status          Show service status"
    echo "  cleanup         Remove all containers and images"
    echo "  help            Show this help message"
    echo
    echo "Environment Variables:"
    echo "  SKIP_HEALTH_CHECK    Skip health check during deployment"
    echo
    echo "Examples:"
    echo "  $0 deploy           # Deploy the application"
    echo "  $0 start            # Start services"
    echo "  $0 logs             # View logs"
    echo "  SKIP_HEALTH_CHECK=true $0 deploy  # Deploy without health check"
}

# Function to start services
start() {
    print_status "Starting services..."
    docker-compose up -d
    print_success "Services started"
}

# Function to stop services
stop() {
    print_status "Stopping services..."
    docker-compose down
    print_success "Services stopped"
}

# Function to restart services
restart() {
    print_status "Restarting services..."
    docker-compose restart
    print_success "Services restarted"
}

# Function to show logs
show_logs() {
    print_status "Showing application logs..."
    docker-compose logs -f
}

# Function to show status
show_status() {
    print_status "Service status:"
    docker-compose ps
}

# Function to cleanup everything
cleanup() {
    print_warning "This will remove all containers, images, and volumes. Are you sure? (y/N)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        print_status "Cleaning up..."
        docker-compose down -v --rmi all
        docker system prune -f
        print_success "Cleanup completed"
    else
        print_status "Cleanup cancelled"
    fi
}

# Main script logic
case "${1:-deploy}" in
    deploy)
        deploy
        ;;
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        restart
        ;;
    logs)
        show_logs
        ;;
    status)
        show_status
        ;;
    cleanup)
        cleanup
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown option: $1"
        show_help
        exit 1
        ;;
esac
