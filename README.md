# SecureSight_AI_Desktop_Application
# SecureSight AI
 
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
 
**AI-Powered Intelligent Surveillance Desktop Application**
 
SecureSight AI is a real-time video surveillance system that uses deep learning (CNN-LSTM) to detect anomalous/criminal activities from multiple camera feeds and sends instant WhatsApp alerts.
 
## Features
 
- **AI Crime Detection**: CNN-LSTM model analyzes video clips in real-time to detect suspicious activities
- **Multi-Camera Support**: Monitor multiple camera feeds simultaneously in a responsive grid layout
- **Instant Alerts**: WhatsApp notifications via Twilio when crimes are detected
- **Video Recording**: Record camera feeds with one click
- **Auto Snapshots**: Automatically capture images when anomalies are detected
- **Modern Dark UI**: Clean, professional interface built with Tkinter
- **Fullscreen View**: Expand any camera feed to fullscreen
- **Persistent Settings**: Camera configurations and preferences saved between sessions
 
## Tech Stack
 
- **GUI**: Tkinter (Python)
- **Deep Learning**: PyTorch, TorchVision
- **Computer Vision**: OpenCV
- **Image Processing**: Pillow (PIL)
- **Notifications**: Twilio WhatsApp API
- **Model**: CNN-LSTM architecture for spatiotemporal feature extraction
 
## Installation
 
```bash
# Clone the repository
git clone https://github.com/yourusername/SecureSight_AI_Desktop_Application.git
cd SecureSight_AI_Desktop_Application
 
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
 
# Install dependencies
pip install -r requirements.txt
Usage
bash
# Run the application
python secure_sight.py
Or use the provided run script:

bash
./run
Configuration
Twilio WhatsApp Alerts (Optional)
Set environment variables before running:

bash
set SECURESIGHT_TWILIO_SID=your_account_sid
set SECURESIGHT_TWILIO_TOKEN=your_auth_token
set SECURESIGHT_TWILIO_FROM=whatsapp:+14155238886
Or edit directly in secure_sight.py (lines 20-22).

User Profile
On first launch, enter your name and mobile number to receive alerts.

Project Structure
SecureSight_AI_Desktop_Application/
├── secure_sight.py       # Main application
├── best_model.pth        # Pre-trained CNN-LSTM model
├── requirements.txt      # Python dependencies
├── icon.ico              # App icon
├── snapshots/            # Auto-captured images
├── securesight_prefs.json # User preferences
└── run                   # Quick launch script
Model Architecture
CNN: Feature extraction from video frames (3 conv layers)
LSTM: Temporal sequence modeling (1 layer, 256 hidden units)
Input: 16-frame clips resized to 64x64
Output: Binary classification (Normal vs Crime)
Screenshots
[Add screenshots of your application here]

License
This project is licensed under the MIT License - see LICENSE file.

Acknowledgments
PyTorch team for the deep learning framework
OpenCV community for computer vision tools
Twilio for WhatsApp messaging API
