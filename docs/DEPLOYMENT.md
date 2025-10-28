# LUMEN TTMS - Deployment Guide

## Production Deployment

This guide covers deployment of the LUMEN TimeTable Management System for production use.

## Deployment Options

### Option 1: Standalone Server Deployment

**Requirements**:
- Ubuntu 20.04+ or similar Linux distribution
- PostgreSQL 12+
- Python 3.8+
- Node.js 14+
- 8GB RAM minimum (16GB+ recommended for large institutions)
- 50GB disk space

**Installation Steps**:

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y postgresql postgresql-contrib python3.8 python3-pip nodejs npm nginx

# Create application user
sudo adduser lumen --disabled-password

# Clone repository
sudo -u lumen git clone https://github.com/snchakri/lumen_25028_ttms.git /home/lumen/app
cd /home/lumen/app

# Setup database
sudo -u postgres createdb lumen_ttms
sudo -u postgres psql -d lumen_ttms -f db_sql_files/hei_timetabling_datamodel.sql
sudo -u postgres psql -d lumen_ttms -f db_sql_files/management-system-schema.sql

# Install Python dependencies
sudo -u lumen python3 -m pip install -r scheduling_system/requirements.txt
sudo -u lumen python3 -m pip install -r test_suite_generator/requirements.txt

# Install Node.js dependencies (for web app)
cd web_app && npm install

# Configure environment variables
cp .env.example .env
# Edit .env with your configuration

# Start services
sudo systemctl start lumen_scheduling
sudo systemctl start lumen_webapp
sudo systemctl enable lumen_scheduling lumen_webapp
```

### Option 2: Docker Deployment

**Requirements**:
- Docker 20.10+
- Docker Compose 1.29+

**Installation Steps**:

```bash
# Clone repository
git clone https://github.com/snchakri/lumen_25028_ttms.git
cd lumen_25028_ttms

# Create environment file
cp .env.example .env
# Edit .env with your configuration

# Build and start containers
docker-compose up -d

# Initialize database
docker-compose exec postgres psql -U lumen_user -d lumen_ttms -f /sql/hei_timetabling_datamodel.sql
docker-compose exec postgres psql -U lumen_user -d lumen_ttms -f /sql/management-system-schema.sql

# Verify deployment
docker-compose ps
docker-compose logs -f
```

**docker-compose.yml**:

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:14
    environment:
      POSTGRES_DB: lumen_ttms
      POSTGRES_USER: lumen_user
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./db_sql_files:/sql
    ports:
      - "5432:5432"
    restart: unless-stopped

  scheduling_engine:
    build:
      context: ./scheduling_system
      dockerfile: Dockerfile
    environment:
      DATABASE_URL: postgresql://lumen_user:${DB_PASSWORD}@postgres:5432/lumen_ttms
      LOG_LEVEL: INFO
    volumes:
      - ./scheduling_system/input:/app/input
      - ./scheduling_system/output:/app/output
      - ./scheduling_system/logs:/app/logs
    depends_on:
      - postgres
    restart: unless-stopped

  webapp:
    build:
      context: ./web_app
      dockerfile: Dockerfile
    environment:
      DATABASE_URL: postgresql://lumen_user:${DB_PASSWORD}@postgres:5432/lumen_ttms
      JWT_SECRET: ${JWT_SECRET}
      NODE_ENV: production
    ports:
      - "3000:3000"
    depends_on:
      - postgres
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./web_app/build:/usr/share/nginx/html:ro
    ports:
      - "80:80"
      - "443:443"
    depends_on:
      - webapp
    restart: unless-stopped

volumes:
  postgres_data:
```

### Option 3: Cloud Deployment (AWS/Azure/GCP)

**AWS Deployment**:

```bash
# Create EC2 instance (t3.medium or larger)
# Install Docker and Docker Compose
# Follow Docker deployment steps above

# For RDS PostgreSQL
# Create RDS instance (PostgreSQL 14)
# Update DATABASE_URL in .env
# Deploy application containers on EC2
```

**Azure Deployment**:
- Azure Container Instances for containerized deployment
- Azure Database for PostgreSQL
- Azure App Service for web application
- Azure Storage for file uploads

**GCP Deployment**:
- Google Compute Engine for VM-based deployment
- Cloud SQL for PostgreSQL
- Google Kubernetes Engine for container orchestration
- Cloud Storage for file management

## Configuration

### Environment Variables

Create `.env` file in project root:

```env
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=lumen_ttms
DB_USER=lumen_user
DB_PASSWORD=secure_password_here

# Application Configuration
APP_ENV=production
LOG_LEVEL=INFO
DEBUG=false

# Web Application
JWT_SECRET=your_jwt_secret_here
SESSION_TIMEOUT=3600
API_PORT=3000

# Scheduling Engine
SOLVER_TIME_LIMIT=300
DEFAULT_SOLVER=cbc
ENABLE_FALLBACK=true
MAX_RETRIES=3

# File Storage
INPUT_DIR=/data/input
OUTPUT_DIR=/data/output
LOG_DIR=/data/logs

# Email Configuration (optional)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@example.com
SMTP_PASSWORD=your_email_password

# Security
ALLOWED_HOSTS=yourdomain.com,www.yourdomain.com
CORS_ORIGINS=https://yourdomain.com
```

### Nginx Configuration

Create `nginx/nginx.conf`:

```nginx
server {
    listen 80;
    server_name yourdomain.com;

    # Redirect to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com;

    # SSL Configuration
    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;

    # Frontend (React build)
    location / {
        root /usr/share/nginx/html;
        try_files $uri $uri/ /index.html;
    }

    # API proxy
    location /api/ {
        proxy_pass http://webapp:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    # File upload size
    client_max_body_size 50M;
}
```

## Database Management

### Backup Strategy

**Automated Backups**:

```bash
# Create backup script
cat > /usr/local/bin/lumen_backup.sh << 'EOF'
#!/bin/bash
BACKUP_DIR=/backups/lumen_ttms
DATE=$(date +%Y%m%d_%H%M%S)
pg_dump -U lumen_user lumen_ttms | gzip > $BACKUP_DIR/lumen_ttms_$DATE.sql.gz
# Keep last 30 days
find $BACKUP_DIR -name "*.sql.gz" -mtime +30 -delete
EOF

chmod +x /usr/local/bin/lumen_backup.sh

# Schedule daily backups
crontab -e
# Add: 0 2 * * * /usr/local/bin/lumen_backup.sh
```

### Database Migrations

```bash
# Version tracking
psql -U lumen_user -d lumen_ttms -c "SELECT * FROM schema_version ORDER BY version_id DESC LIMIT 1;"

# Apply migration
psql -U lumen_user -d lumen_ttms -f migrations/001_add_phone_number.sql
```

## Monitoring & Logging

### Application Logs

Logs are stored in:
- Scheduling Engine: `/data/logs/scheduling/`
- Web Application: `/data/logs/webapp/`
- System Logs: `/var/log/lumen/`

**Log Rotation**:

```bash
# /etc/logrotate.d/lumen
/data/logs/scheduling/*.log /data/logs/webapp/*.log {
    daily
    rotate 30
    compress
    delaycompress
    notifempty
    create 0640 lumen lumen
    sharedscripts
    postrotate
        systemctl reload lumen_scheduling lumen_webapp
    endscript
}
```

### Performance Monitoring

**System Metrics**:
```bash
# CPU and Memory
htop

# Disk usage
df -h
du -sh /data/*

# Database connections
psql -U lumen_user -d lumen_ttms -c "SELECT count(*) FROM pg_stat_activity;"

# Application health
curl http://localhost:3000/health
```

## Security

### SSL/TLS Configuration

**Using Let's Encrypt**:

```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com

# Auto-renewal (already configured by certbot)
sudo systemctl status certbot.timer
```

### Firewall Configuration

```bash
# UFW firewall
sudo ufw allow 22/tcp   # SSH
sudo ufw allow 80/tcp   # HTTP
sudo ufw allow 443/tcp  # HTTPS
sudo ufw enable
```

### Database Security

```bash
# Restrict PostgreSQL access
# Edit /etc/postgresql/14/main/pg_hba.conf
# Add:
# host    lumen_ttms    lumen_user    localhost    md5

# Restart PostgreSQL
sudo systemctl restart postgresql
```

## Maintenance

### Routine Tasks

**Weekly**:
- Check disk space
- Review error logs
- Verify backup integrity
- Update security patches

**Monthly**:
- Database vacuum and analyze
- Review user access logs
- Update dependencies
- Performance tuning

**Quarterly**:
- Full system backup
- Security audit
- Capacity planning review
- Update documentation

### Troubleshooting

**Service not starting**:
```bash
# Check service status
sudo systemctl status lumen_scheduling
sudo journalctl -u lumen_scheduling -n 50

# Check logs
tail -n 100 /data/logs/scheduling/error.log
```

**Database connection issues**:
```bash
# Test database connection
psql -U lumen_user -h localhost -d lumen_ttms

# Check PostgreSQL status
sudo systemctl status postgresql
sudo -u postgres psql -c "SELECT version();"
```

**Performance issues**:
```bash
# Check system resources
free -h
top
iostat

# Database performance
psql -U lumen_user -d lumen_ttms -c "SELECT * FROM pg_stat_activity;"
```

## Scaling

### Horizontal Scaling

- Load balancer (Nginx/HAProxy)
- Multiple application server instances
- Database replication (read replicas)
- Distributed file storage

### Vertical Scaling

- Increase CPU cores
- Add more RAM
- Upgrade to SSD storage
- Optimize database indexes

## Support & Updates

### Updating the System

```bash
# Backup before update
/usr/local/bin/lumen_backup.sh

# Pull latest code
cd /home/lumen/app
sudo -u lumen git pull origin main

# Update dependencies
sudo -u lumen pip install -r scheduling_system/requirements.txt --upgrade

# Restart services
sudo systemctl restart lumen_scheduling lumen_webapp
```

### Rolling Back

```bash
# Restore database from backup
gunzip -c /backups/lumen_ttms_20251028_020000.sql.gz | psql -U lumen_user lumen_ttms

# Revert code
cd /home/lumen/app
sudo -u lumen git checkout <previous_commit_hash>
sudo systemctl restart lumen_scheduling lumen_webapp
```

---

**For production support**: Refer to system documentation in `docs/` folder  
**For technical issues**: Check logs and monitoring dashboards  
**For security updates**: Subscribe to security mailing list
