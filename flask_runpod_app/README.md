# Flask RunPod Pose Analysis Application

A production-ready Flask web application that leverages RunPod GPU infrastructure for advanced 3D pose analysis using MediaPipe and SMPL-X models.

## Features

- 🎥 **Video Upload**: Drag-and-drop interface for MP4 videos (up to 5GB)
- 🚀 **GPU Processing**: On-demand RunPod A5000 GPU for fast processing
- 📊 **3D Pose Analysis**: MediaPipe → SMPL-X fitting → Ergonomic analysis
- 📈 **Real-time Progress**: Server-Sent Events (SSE) for live updates
- 💾 **Cloud Storage**: Cloudflare R2 for file storage with 7-day retention
- 📧 **Email Notifications**: Automatic notifications when processing completes
- 👥 **Multi-user Support**: 10 internal users with authentication
- 📱 **Responsive Design**: Modern UI with Tailwind CSS

## Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd flask_runpod_app

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your credentials
nano .env  # or use your preferred editor
```

Required configurations:
- **RunPod**: API key and Pod ID
- **Cloudflare R2**: Account ID, access keys, bucket name
- **Email**: SMTP credentials (Gmail or SendGrid)

### 3. Initialize Database

```bash
# Create database and initial users
flask init-db

# Or using Python
python -c "from app import app, db, init_auth; app.app_context().push(); db.create_all(); init_auth(app)"
```

### 4. Run Development Server

```bash
# Development mode
python app.py

# Or with Flask CLI
flask run --host=0.0.0.0 --port=5000

# Production mode with Gunicorn
gunicorn app:app --workers 2 --bind 0.0.0.0:5000
```

Access the application at `http://localhost:5000`

## Default Users

The application comes with 10 pre-configured users:

| Username | Password | Role |
|----------|----------|------|
| admin | admin123 | Admin |
| user1-8 | user123 | User |
| demo | demo123 | User |

## Configuration Guide

### RunPod Setup

1. **Create RunPod Account**: Sign up at [runpod.io](https://runpod.io)
2. **Create Persistent Pod**:
   - GPU: RTX A5000 or better
   - Template: PyTorch 2.0 + CUDA 11.8
   - Storage: 50GB+ persistent volume
3. **Install Dependencies on Pod**:
   ```bash
   # SSH into your RunPod
   cd /workspace
   git clone <main-repo-url> pracovni_poloha_mesh
   cd pracovni_poloha_mesh
   conda create -n trunk_analysis python=3.9 -y
   conda activate trunk_analysis
   pip install -r requirements_runpod.txt
   ```
4. **Note Pod ID**: Found in RunPod dashboard

### Cloudflare R2 Setup

1. **Create Cloudflare Account**: Sign up at [cloudflare.com](https://cloudflare.com)
2. **Enable R2**:
   - Go to R2 in dashboard
   - Create new bucket (e.g., `pose-analysis-files`)
3. **Generate API Credentials**:
   - Create API token with R2 read/write permissions
   - Note Account ID, Access Key ID, and Secret Access Key
4. **Configure Lifecycle Rules** (optional):
   - Set automatic deletion after 7 days

### Email Configuration

#### Gmail Setup
1. Enable 2-factor authentication
2. Generate app-specific password
3. Use in `.env`:
   ```
   SMTP_SERVER=smtp.gmail.com
   SMTP_PORT=587
   SMTP_USERNAME=your-email@gmail.com
   SMTP_PASSWORD=your-app-password
   ```

#### SendGrid Setup (Alternative)
1. Create SendGrid account (100 emails/day free)
2. Generate API key
3. Use in `.env`:
   ```
   SMTP_SERVER=smtp.sendgrid.net
   SMTP_PORT=587
   SMTP_USERNAME=apikey
   SMTP_PASSWORD=your-sendgrid-api-key
   ```

## Deployment

### Render.com Deployment

1. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Create Render Web Service**:
   - Connect GitHub repository
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
   - Add environment variables from `.env`

3. **Configure Persistent Disk** (for SQLite):
   - Mount path: `/opt/render/project/src/data`
   - Update `DATABASE_URL` in environment variables

### Manual Deployment

For VPS or dedicated server:

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install python3.9 python3-pip nginx supervisor

# Setup application
cd /var/www
git clone <repo-url> flask_runpod_app
cd flask_runpod_app
pip3 install -r requirements.txt

# Configure Nginx (see nginx.conf.example)
sudo cp nginx.conf.example /etc/nginx/sites-available/flask_runpod_app
sudo ln -s /etc/nginx/sites-available/flask_runpod_app /etc/nginx/sites-enabled/
sudo nginx -s reload

# Configure Supervisor (see supervisor.conf.example)
sudo cp supervisor.conf.example /etc/supervisor/conf.d/flask_runpod_app.conf
sudo supervisorctl reread
sudo supervisorctl update
```

## API Endpoints

### Public Endpoints
- `GET /` - Home page
- `GET /login` - Login page
- `POST /login` - Authenticate user
- `GET /health` - Health check

### Authenticated Endpoints
- `GET /upload` - Upload page
- `POST /api/upload` - Upload video
- `GET /progress/<job_id>` - Progress page
- `GET /api/progress/<job_id>` - SSE progress stream
- `GET /api/job/<job_id>` - Job status
- `GET /api/job/<job_id>/files` - Download links
- `GET /history` - Job history

### Admin Endpoints
- `GET /admin/dashboard` - Admin dashboard
- `GET /admin/logs` - System logs
- `GET /api/test/email` - Test email configuration
- `GET /api/test/runpod` - Test RunPod connection
- `GET /api/test/storage` - Test R2 storage

## Testing

### Run Tests
```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/

# With coverage
pytest tests/ --cov=.
```

### Manual Testing
1. **Test Upload**: Upload a small test video
2. **Test Processing**: Monitor progress in real-time
3. **Test Downloads**: Verify PKL and XLSX files
4. **Test Email**: Check notifications

### Test Services
```bash
# Test all services
flask test-services

# Or individual tests
curl http://localhost:5000/api/test/runpod
curl http://localhost:5000/api/test/storage
curl http://localhost:5000/api/test/email
```

## Troubleshooting

### Common Issues

1. **RunPod Connection Failed**
   - Verify API key and Pod ID
   - Check if Pod is running
   - Ensure SSH key is configured (if using SSH)

2. **R2 Upload Failed**
   - Verify Cloudflare credentials
   - Check bucket exists and is accessible
   - Ensure proper CORS configuration

3. **Email Not Sending**
   - Verify SMTP credentials
   - Check firewall for port 587/465
   - Enable "Less secure apps" for Gmail

4. **Database Errors**
   - Run `flask init-db` to reinitialize
   - Check file permissions for SQLite
   - Verify DATABASE_URL in environment

5. **SSE Not Working**
   - Check reverse proxy configuration
   - Disable buffering in Nginx
   - Verify JavaScript console for errors

### Debug Mode

Enable debug logging:
```python
# In app.py
app.config['DEBUG'] = True
logging.basicConfig(level=logging.DEBUG)
```

View logs:
```bash
tail -f logs/app.log
```

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Browser   │────▶│  Flask App   │────▶│  RunPod GPU │
│   (User)    │◀────│  (Render)    │◀────│  (A5000)    │
└─────────────┘     └──────────────┘     └─────────────┘
                            │                      │
                            ▼                      ▼
                    ┌──────────────┐     ┌──────────────┐
                    │  SQLite DB   │     │ Cloudflare R2│
                    │  (Jobs/Logs) │     │  (Storage)   │
                    └──────────────┘     └──────────────┘
```

## Cost Analysis

| Service | Usage | Monthly Cost |
|---------|-------|--------------|
| Render.com | 750 hrs free | $0 |
| Cloudflare R2 | 10GB free | $0 |
| RunPod A5000 | ~10 hrs @ $0.79/hr | ~$8 |
| Email (Gmail) | 500/day free | $0 |
| **Total** | | **~$8/month** |

Per-video cost: ~$0.40 (30-minute video)

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
- GitHub Issues: [Create an issue](https://github.com/your-repo/issues)
- Email: support@example.com

## Acknowledgments

- MediaPipe team for pose detection
- SMPL-X team for 3D human model
- RunPod for GPU infrastructure
- Cloudflare for R2 storage