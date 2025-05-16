# Mock Data Generation for YC Demo

This directory contains scripts to generate mock data for the Sift dashboard YC demo.

## Features

- Generates realistic camera data
- Generates safety violation alerts with various violation types
- Creates mock screenshots for alerts
- Uses weighted distribution to create realistic patterns (more alerts during work hours, etc.)

## Using with Docker

The mock data generation is configured to run automatically when you start the Docker containers. 

### Quick Start

1. Start the application with mock data generation:

```bash
# By default, mock data generation is enabled
docker-compose up -d
```

2. Browse to the frontend (typically http://localhost:3000) to see the populated dashboard.

### Customizing Mock Data Generation

You can control whether mock data is generated using environment variables:

- Set `GENERATE_MOCK_DATA=true` to enable mock data generation (default)
- Set `GENERATE_MOCK_DATA=false` to disable mock data generation

Example:
```bash
GENERATE_MOCK_DATA=false docker-compose up -d
```

## Running Scripts Manually

If you need to run the scripts manually, you can do so in the Docker container:

```bash
# Execute the mock data generation script in the running container
docker exec -it sift-mvp-backend-1 python /app/backend/scripts/generate_mock_data.py

# Execute the mock screenshots generation script
docker exec -it sift-mvp-backend-1 python /app/backend/scripts/generate_mock_screenshots.py
```

Or run the convenience script that does both:

```bash
docker exec -it sift-mvp-backend-1 /app/backend/scripts/setup_mock_data.sh
```

## Configuration

You can modify the following constants in `generate_mock_data.py` to customize the mock data:

- `NUM_CAMERAS`: Number of cameras to generate (default: 8)
- `NUM_ALERTS`: Total number of alerts to generate (default: 200)
- `HOURS_BACK`: Generate data for the past N hours (default: 72)
- `TIME_VARIANCE`: Make more alerts during "work hours" if True (default: True)

## Troubleshooting

If the dashboard is not showing any data:

1. Check if the database is running:
```bash
docker-compose ps
```

2. Verify mock data was generated:
```bash
docker exec -it sift-mvp-backend-1 python -c "from backend.database import SessionLocal; db = SessionLocal(); from backend.models import Alert; print(f'Alerts in DB: {db.query(Alert).count()}'); db.close()"
```

3. If needed, run the mock data generation manually:
```bash
docker exec -it sift-mvp-backend-1 python /app/backend/scripts/generate_mock_data.py
docker exec -it sift-mvp-backend-1 python /app/backend/scripts/generate_mock_screenshots.py
``` 