# AI-Enabled GUI for Medical Image Analysis![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)![Python](https://img.shields.io/badge/python-3.9+-green.svg)![License](https://img.shields.io/badge/license-MIT-orange.svg)![Status](https://img.shields.io/badge/status-active-brightgreen.svg)## 📋 Table of Contents- [Overview](#overview)- [Features](#features)- [Architecture](#architecture)- [Installation](#installation)- [Docker Deployment](#docker-deployment)- [Usage](#usage)- [Authentication](#authentication)- [API Reference](#api-reference)- [Configuration](#configuration)- [Troubleshooting](#troubleshooting)- [Contributing](#contributing)- [License](#license)## 🏥 OverviewThe **AI-Enabled GUI for Medical Image Analysis** is an advanced web-based platform that combines cutting-edge AI models for precise medical image segmentation. Built with Gradio and powered by SAM (Segment Anything Model) and GDINO (Grounding DINO), this application provides medical professionals with intuitive tools for analyzing medical images with AI assistance.### 🎯 Key Capabilities- **Text-guided Segmentation**: Use natural language descriptions to identify anatomical structures- **Interactive Point Annotations**: Click-based region of interest marking- **Bounding Box Guidance**: Precise area-focused analysis- **Multi-modal Analysis**: Combine text, points, and bounding boxes for optimal results- **Role-based Access Control**: Secure multi-user environment for medical teams- **Audit Logging**: Complete traceability of all analysis activities## ✨ Features### 🔐 Security & Authentication- **Multi-role User Management**: Administrator, Radiologist, Researcher, Guest roles- **Session-based Authentication**: Secure 8-hour session management- **Permission Matrix**: Granular access control for different user types- **Audit Trail**: Complete logging of user activities and analysis results### 🤖 AI-Powered Analysis- **SAM 2.1 Integration**: State-of-the-art segmentation capabilities- **GDINO Object Detection**: Text-guided object detection and localization- **Custom Medical Models**: Brain tumor detection and specialized medical imaging- **Multiple Analysis Modes**: Text prompt, point annotations, bounding box, and combined approaches### 🖥️ User Interface- **Responsive Design**: Modern, clinical-grade interface optimized for medical workflows- **Real-time Feedback**: Live annotation and analysis status updates- **Multi-format Support**: JPEG, PNG image processing- **Professional Styling**: Medical-themed UI with intuitive navigation### 📊 Data Management- **Automated Saving**: Results automatically saved with user attribution- **Organized Storage**: Structured file organization with timestamps- **Export Capabilities**: Easy data export for research and clinical documentation- **Image History**: Track all processed images and their results## 🏗️ Architecture```┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐│   Web Interface │    │   Authentication │    │   AI Processing ││   (Gradio GUI)  │◄──►│     System       │◄──►│   (SAM + GDINO) │└─────────────────┘    └──────────────────┘    └─────────────────┘         │                        │                        │         ▼                        ▼                        ▼┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐│ User Management │    │  Session Storage │    │  Result Storage ││   & Permissions │    │   & Audit Logs   │    │   & File Mgmt   │└─────────────────┘    └──────────────────┘    └─────────────────┘```### Core Components1. **Frontend**: Gradio-based web interface with clinical design2. **Backend**: LitServe API for model inference and processing3. **Authentication**: Role-based access control with session management4. **AI Models**: SAM 2.1 for segmentation, GDINO for object detection5. **Storage**: Organized file system for images and results## 🚀 Installation### Prerequisites- Python 3.9 or higher- CUDA-compatible GPU (recommended for optimal performance)- 8GB+ RAM recommended- 5GB+ free disk space### Local Installation1. **Clone the Repository**   ```bash   git clone <repository-url>   cd AIEnabledImageAnnotation   ```2. **Create Virtual Environment**   ```bash   python -m venv venv      # Windows   venv\Scripts\activate      # Linux/macOS   source venv/bin/activate   ```3. **Install Dependencies**   ```bash   pip install -r requirements.txt   ```4. **Set Up Directory Structure**   ```bash   mkdir -p images/input images/annotated   ```5. **Initialize Authentication**   ```bash   # The application will create default users on first run   # Default admin credentials: admin/admin123   ```6. **Run the Application**   ```bash   python app_secure.py   ```7. **Access the Application**   - Open your browser and navigate to: `http://localhost:8000/aisegmentation`## 🐳 Docker Deployment### Quick Start with Docker1. **Build the Docker Image**   ```bash   docker build -t ai-medical-analysis .   ```2. **Run the Container**   ```bash   docker run -p 8000:8000 -v $(pwd)/images:/app/images ai-medical-analysis   ```3. **Access the Application**   - Navigate to: `http://localhost:8000/aisegmentation`### Docker Compose (Recommended)1. **Create docker-compose.yml**   ```yaml   version: '3.8'   services:     ai-medical-analysis:       build: .       ports:         - "8000:8000"       volumes:         - ./images:/app/images         - ./users.json:/app/users.json       environment:         - PYTHONUNBUFFERED=1       restart: unless-stopped   ```2. **Deploy with Docker Compose**   ```bash   docker-compose up -d   ```### Production DeploymentFor production environments, consider:- Using a reverse proxy (nginx)- Implementing SSL/TLS certificates- Setting up proper logging and monitoring- Configuring backup strategies for user data and results## 🔧 Usage### Getting Started1. **Login**: Use your medical professional credentials   - **Administrator**: Full system access and user management   - **Radiologist**: Medical analysis with full result access   - **Researcher**: Research-focused analysis capabilities   - **Guest**: Limited read-only access2. **Upload Medical Image**: Support for JPEG and PNG formats3. **Choose Analysis Mode**:   - **Text Prompt**: Use medical terminology (e.g., "brain tumor", "lung nodule")   - **Point Annotations**: Click directly on regions of interest   - **Bounding Box**: Define specific areas for focused analysis   - **Combined Modes**: Use multiple approaches for optimal results4. **Configure Parameters**:   - **SAM Model**: Choose appropriate model for your use case   - **Detection Threshold**: Adjust sensitivity for feature detection   - **Text Matching Threshold**: Fine-tune confidence for text-based detection5. **Analyze**: Click "Analyze Medical Image" to process6. **Review Results**: Examine segmentation results and save for documentation### Medical Use Cases**Brain Imaging**```Text Prompt: "brain tumor"Bounding Box: Focus on suspected tumor regionCombined: Use text + bounding box for precise tumor segmentation```**Chest X-rays**```Text Prompt: "lung nodule"Point Annotations: Click on suspicious areasCombined: Text + points for comprehensive lung analysis```**Cardiac Imaging**```Text Prompt: "heart ventricle"Bounding Box: Define cardiac regionAnalysis: Automatic ventricle segmentation and measurement```## 🔐 Authentication### Default UsersThe system creates default users on first startup:| Username   | Password     | Role          | Capabilities ||------------|--------------|---------------|--------------|| admin      | admin123     | Administrator | Full system access, user management || doctor     | doctor123    | Radiologist   | Medical analysis, view all results || researcher | research123  | Researcher    | Research analysis, limited access |### Permission Matrix| Permission         | Admin | Radiologist | Researcher | Guest ||-------------------|-------|-------------|------------|-------|| Analyze Images    | ✅     | ✅           | ✅          | ❌     || View All Results  | ✅     | ✅           | ❌          | ❌     || Manage Users      | ✅     | ❌           | ❌          | ❌     || Export Data       | ✅     | ✅           | ❌          | ❌     || Modify Settings   | ✅     | ❌           | ❌          | ❌     || View System Logs  | ✅     | ❌           | ❌          | ❌     |### Creating New UsersAdministrators can create new users through the Admin Controls panel:1. Navigate to "Admin Controls" → "User Management"2. Fill in user details (username, password, role, full name, email)3. Click "Create User"4. User receives access based on assigned role permissions## 📡 API Reference### Authentication Endpoints#### Login```pythonPOST /auth/login{    "username": "string",    "password": "string"}```#### Logout```pythonPOST /auth/logout{    "session_id": "string"}```### Analysis Endpoints#### Image Analysis```pythonPOST /predictContent-Type: multipart/form-dataFiles:- image: Medical image file (JPEG/PNG)Data:- sam_type: SAM model identifier- box_threshold: Detection confidence threshold- text_threshold: Text matching threshold- text_prompt: Medical description- user: Username- role: User role- bounding_box: Optional coordinates (x1,y1,x2,y2)- point_prompts: Optional point coordinates- point_labels: Optional point labels```#### Response Format```python{    "status": "success",    "image": "base64_encoded_result",    "analysis_info": {        "detected_objects": [...],        "segmentation_masks": [...],        "confidence_scores": [...]    }}```## ⚙️ Configuration### Environment VariablesCreate a `.env` file for custom configuration:```bash# Server ConfigurationPORT=8000HOST=0.0.0.0# Model ConfigurationDEFAULT_SAM_MODEL=sam2.1_hiera_smallMODEL_CACHE_DIR=./models# Security ConfigurationSESSION_TIMEOUT=28800  # 8 hours in secondsMAX_LOGIN_ATTEMPTS=5LOCKOUT_DURATION=900   # 15 minutes# File StorageUPLOAD_MAX_SIZE=50     # MBRESULTS_RETENTION=30   # days# LoggingLOG_LEVEL=INFOLOG_FILE=medical_analysis.log```### Model Configuration

The application supports multiple SAM models:
- `sam2.1_hiera_small` (default - fastest)
- `sam2.1_hiera_base` (balanced performance)
- `sam2.1_hiera_large` (highest accuracy)
- `brain_tumor_sam_vit_base` (specialized medical model)

Configure in `sam_libs/server.py`:
```python
self.model = LangSAM("your_preferred_model")
```

## 🔍 Troubleshooting

### Common Issues

**1. Login Issues**
```bash
# Reset to default users
rm users.json
python app_secure.py
```

**2. Model Loading Errors**
```bash
# Verify CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Clear model cache
rm -rf ~/.cache/torch/hub/
```

**3. Memory Issues**
```bash
# Monitor GPU memory
nvidia-smi

# Reduce batch size or use smaller model
# Edit sam_libs/server.py to use sam2.1_hiera_small
```

**4. Port Already in Use**
```bash
# Find process using port 8000
netstat -tulpn | grep :8000

# Kill process or change port in configuration
```

### Logs and Debugging

- Application logs: Check console output during startup
- Authentication logs: User login/logout activities are logged
- Analysis logs: All image processing activities are tracked
- Error logs: Detailed error information for troubleshooting

### Performance Optimization

**For CPU-only Systems:**
```python
# In sam_libs/server.py, modify setup method:
def setup(self, device: str) -> None:
    self.model = LangSAM("sam2.1_hiera_small", device="cpu")
```

**For High-Performance Systems:**
```python
# Use larger model for better accuracy:
def setup(self, device: str) -> None:
    self.model = LangSAM("sam2.1_hiera_large", device=device)
```

## 🤝 Contributing

We welcome contributions to improve the AI-Enabled GUI for Medical Image Analysis!

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Install development dependencies: `pip install -r requirements-dev.txt`
4. Make your changes and add tests
5. Commit your changes: `git commit -m 'Add amazing feature'`
6. Push to the branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

### Code Standards

- Follow PEP 8 style guide
- Add docstrings to all functions and classes
- Include unit tests for new features
- Update documentation as needed

### Reporting Issues

Please use the GitHub issue tracker to report bugs or request features:
- Provide detailed description of the issue
- Include steps to reproduce
- Specify your environment (OS, Python version, etc.)
- Attach relevant log files or screenshots

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **SAM (Segment Anything Model)** by Meta AI Research
- **GDINO (Grounding DINO)** for object detection capabilities
- **Gradio** for the intuitive web interface framework
- **LitServe** for efficient model serving infrastructure
- The medical AI research community for advancing healthcare technology

## 📞 Support

For support and questions:
- 📧 Email: support@medical-ai.com
- 📖 Documentation: [Wiki](repository-wiki-url)
- 💬 Discussions: [GitHub Discussions](repository-discussions-url)
- 🐛 Bug Reports: [GitHub Issues](repository-issues-url)

---

**⚠️ Medical Disclaimer**: This software is for research and educational purposes. It should not be used as a substitute for professional medical diagnosis. Always consult qualified healthcare professionals for medical decisions.

**🔒 Privacy Notice**: This application processes medical images locally. Ensure compliance with applicable healthcare privacy regulations (HIPAA, GDPR, etc.) in your jurisdiction.
