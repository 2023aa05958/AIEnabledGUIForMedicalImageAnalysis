# Build and Deployment Guide

This folder contains all build-related files for the AI-Enabled Medical Image Analysis application.

## Folder Structure

```
build/
├── docker/                 # Docker development files
│   ├── Dockerfile          # Container definition
│   ├── docker-compose.yml  # Multi-container orchestration  
│   └── .dockerignore       # Docker ignore rules
├── scripts/                # Quick start scripts  
│   ├── run_app.bat         # Windows startup script
│   └── run_app.sh          # Linux startup script
└── deployment/             # Production deployment automation
    ├── deploy.bat          # Windows deployment automation
    └── deploy.sh           # Linux deployment automation
```

## Purpose of Each Folder

- **🐳 `docker/`** - Docker containers and compose files for development
- **⚡ `scripts/`** - Quick development startup scripts
- **🚀 `deployment/`** - Production deployment automation scripts

## Quick Start

### Development Mode
**Windows:**
```cmd
build\scripts\run_app.bat
```

**Linux:**
```bash
build/scripts/run_app.sh
```

### Docker Development
```bash
cd build/docker
docker-compose up --build
```

### Production Deployment
**Windows:**
```cmd
cd build\deployment
deploy.bat production
```

**Linux:**
```bash
cd build/deployment
./deploy.sh production
```

## Build Process

1. **Dependencies**: All requirements are in `config/requirements.txt`
2. **Source Code**: Located in `src/` directory
   - Applications: `src/apps/`
   - Authentication: `src/auth/`
   - Libraries: `src/libs/sam_libs/`
3. **Data**: Application data stored in `data/` directory
4. **Models**: SAM models in `src/libs/sam_libs/models/`

## Docker Build Context

The Docker build context is set to the project root (`../../`) to access all necessary files while keeping build files organized in this dedicated folder.

## Notes

- All build scripts automatically navigate to the correct project root
- Docker files use relative paths from the project root
- Environment configuration is in `config/` directory
- Build artifacts and logs are stored in project root `logs/` folder
- SAM libraries are now properly organized in `src/libs/sam_libs/`
