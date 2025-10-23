# FSE4AI-2025 Final Project - AI Image Classifier

A web-based image classification application using FastAPI and PyTorch MobileNetV2. Upload an image and get instant predictions from a pre-trained neural network.

## Features

- 🖼️ Real-time image classification using MobileNetV2
- 🌐 Clean and responsive web interface
- 🐳 Docker support for easy deployment
- ⚙️ CI/CD pipeline with GitHub Actions
- 🚀 FastAPI backend for high performance
- 📦 Lightweight CPU-optimized PyTorch

## Prerequisites

- Python 3.10+
- Docker (optional, for containerized deployment)
- Git

## Installation & Setup

### Method 1: Local Installation (Python Virtual Environment)

1. **Clone the repository**
```bash
git clone https://github.com/MohamedNguira/FSE4AI-2025-Final-Project.git
cd FSE4AI-2025-Final-Project
```

2. **Create and activate virtual environment**

On Windows:
```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On Linux/Mac:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. **Install dependencies**
```bash
pip install --upgrade pip
pip install fastapi uvicorn[standard] Pillow python-multipart
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
```

Or install from requirements.txt:
```bash
pip install -r requirements.txt
```

4. **Create static directory**
```bash
mkdir -p static
```

5. **Run the application**
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

6. **Open in browser**
Navigate to: http://localhost:8000

### Method 2: Docker (Recommended for Production)

#### Using Docker Compose (Easiest)

1. **Clone the repository**
```bash
git clone https://github.com/MohamedNguira/FSE4AI-2025-Final-Project.git
cd FSE4AI-2025-Final-Project
```

2. **Build and run with Docker Compose**
```bash
docker-compose up --build
```

3. **Access the application**
Navigate to: http://localhost:8000

4. **Stop the application**
```bash
docker-compose down
```

#### Using Docker CLI

1. **Build the Docker image**
```bash
docker build -t fse4ai-classifier .
```

2. **Run the container**
```bash
docker run -d -p 8000:8000 --name image-classifier fse4ai-classifier
```

3. **View logs**
```bash
docker logs -f image-classifier
```

4. **Stop the container**
```bash
docker stop image-classifier
docker rm image-classifier
```

## Project Structure

```
FSE4AI-2025-Final-Project/
├── .github/
│   └── workflows/
│       └── ci.yml              # CI/CD pipeline configuration
├── static/                     # Static files (images for testing)
│   ├── chimp.jpeg
│   └── dog.jpeg
├── tests/                      # Unit tests
│   ├── __init__.py
│   └── test_app.py
├── app.py                      # Main FastAPI application
├── docker-compose.yml          # Docker Compose configuration
├── Dockerfile                  # Docker image definition
├── .dockerignore              # Docker ignore patterns
├── .gitignore                 # Git ignore patterns
├── requirements.txt           # Python dependencies
├── imagenet_classes.txt       # ImageNet class labels (auto-downloaded)
├── LICENSE
└── README.md
```

## Usage

1. Open the web interface at http://localhost:8000
2. Click "Choose File" and select an image from your computer
3. Click "Predict" to classify the image
4. The prediction will appear below the button

### Test Images

Try the application with the sample images in the `static/` folder:
- `chimp.jpeg` - Should classify as a primate
- `dog.jpeg` - Should classify as a dog breed

## CI/CD Pipeline

The project includes automated CI/CD using GitHub Actions:

### Pipeline Stages

1. **Lint** - Code quality checks with Black and Flake8
2. **Test** - Run unit tests with pytest
3. **Docker** - Build and test Docker image

### Triggering the Pipeline

The pipeline runs automatically on:
- Push to `main` or `develop` branches
- Pull requests to `main` branch

### Viewing Pipeline Status

Check the Actions tab in the GitHub repository to see pipeline results.

## Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html
```

## Development

### Creating a Feature Branch

```bash
# Create and switch to a new branch
git checkout -b feature/your-feature-name

# Make your changes
# ...

# Commit and push
git add .
git commit -m "Add your feature"
git push origin feature/your-feature-name

# Create a pull request on GitHub
```

### Code Style

This project uses:
- **Black** for code formatting
- **Flake8** for linting

Format your code before committing:
```bash
black app.py
flake8 app.py --max-line-length=120
```

## API Endpoints

### GET /
Returns the HTML web interface.

### POST /predict/
Accepts an image file and returns classification prediction.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: `file` (image file)

**Response:**
```json
{
  "predicted_class": "golden_retriever"
}
```

## Model Information

- **Architecture**: MobileNetV2
- **Pre-trained on**: ImageNet (1000 classes)
- **Input size**: 224x224 pixels
- **Framework**: PyTorch
- **Optimization**: CPU-optimized version

## Troubleshooting

### Common Issues

**Port 8000 already in use:**
```bash
# Find and kill the process
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:8000 | xargs kill -9
```

**PyTorch installation issues:**
Use CPU-only version:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**Docker image too large:**
The image uses CPU-only PyTorch to reduce size. GPU version would be significantly larger.

**Module not found errors:**
Ensure virtual environment is activated and dependencies are installed:
```bash
pip install -r requirements.txt
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## Team Members

- Mohamed Nguira
- [Add other team members]

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PyTorch for the pre-trained MobileNetV2 model
- FastAPI for the web framework
- ImageNet dataset for model training

## Timeline

- **Deadline**: Friday
- **Final submission**: 5-minute video demonstration

## Contact

For questions or issues, please create an issue on GitHub or contact the team members.

---

**Note**: This is a educational project for FSE4AI-2025 course.
