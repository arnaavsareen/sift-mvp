#!/usr/bin/env python3
"""
Script to generate mock data for Sift dashboard YC demo.
This will create cameras and alerts with realistic data patterns.
"""

import sys
import os
import random
import datetime
from sqlalchemy.orm import Session
from sqlalchemy import func, inspect
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("mock_data_generator")

# Add parent directory to path to import backend modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    # Import models and database
    from backend.database import SessionLocal, engine, Base
    from backend.models import Camera, Alert, Zone
    
    # Configuration
    NUM_CAMERAS = 8
    NUM_ALERTS = 200  # Total number of alerts to generate
    HOURS_BACK = 72  # Generate data for the last 72 hours
    TIME_VARIANCE = True  # Make more alerts during "work hours"
    
    # Lists for generating realistic data
    CAMERA_NAMES = [
        "Building A Entrance", "Loading Dock", "Assembly Line 1", 
        "Packaging Area", "Storage Warehouse", "Construction Site A",
        "Machine Shop", "Building B Entrance", "Welding Station",
        "Chemical Storage", "Maintenance Area", "Shipping Zone"
    ]
    
    LOCATIONS = [
        "North Building", "South Building", "East Wing", 
        "West Wing", "Building A", "Building B",
        "Warehouse", "Manufacturing Floor", "Construction Site"
    ]
    
    VIOLATION_TYPES = [
        "no_hardhat",
        "no_safety_vest",
        "unsafe_distance",
        "restricted_area",
        "improper_lifting",
        "no_safety_goggles"
    ]
    
    # Weighted probabilities for violation types (more common types have higher weights)
    VIOLATION_WEIGHTS = {
        "no_hardhat": 0.35,
        "no_safety_vest": 0.30,
        "unsafe_distance": 0.15,
        "restricted_area": 0.10,
        "improper_lifting": 0.05,
        "no_safety_goggles": 0.05
    }
    
    def wait_for_db(max_retries=30, retry_interval=2):
        """Wait for the database to be available"""
        logger.info("Checking database connection...")
        retries = 0
        while retries < max_retries:
            try:
                # Try to connect to the database
                inspector = inspect(engine)
                tables = inspector.get_table_names()
                logger.info(f"Database connected! Tables found: {tables}")
                return True
            except Exception as e:
                retries += 1
                if retries >= max_retries:
                    logger.error(f"Failed to connect to database after {max_retries} attempts: {e}")
                    return False
                logger.warning(f"Database not available yet. Retrying in {retry_interval}s... ({retries}/{max_retries})")
                time.sleep(retry_interval)
        return False
    
    def generate_cameras(db: Session, num_cameras: int):
        """Generate mock camera data"""
        logger.info(f"Generating {num_cameras} cameras...")
        
        # Delete existing cameras (will cascade delete alerts)
        db.query(Camera).delete()
        db.commit()
        
        cameras = []
        used_names = set()
        
        for i in range(num_cameras):
            # Ensure unique camera names
            name = random.choice([n for n in CAMERA_NAMES if n not in used_names])
            used_names.add(name)
            
            # If we've used all names, append a number to create new ones
            if len(used_names) >= len(CAMERA_NAMES):
                name = f"{random.choice(CAMERA_NAMES)} {len(used_names) - len(CAMERA_NAMES) + 1}"
            
            camera = Camera(
                name=name,
                url=f"rtsp://example.com/stream{i + 1}",
                location=random.choice(LOCATIONS),
                is_active=random.random() > 0.2,  # 80% are active
                created_at=datetime.datetime.now() - datetime.timedelta(days=random.randint(10, 30))
            )
            db.add(camera)
        
        db.commit()
        
        # Fetch all cameras we just created
        cameras = db.query(Camera).all()
        logger.info(f"Generated {len(cameras)} cameras")
        return cameras
    
    def generate_weighted_violation():
        """Generate a violation type based on weighted probabilities"""
        r = random.random()
        cumulative = 0
        for violation, weight in VIOLATION_WEIGHTS.items():
            cumulative += weight
            if r <= cumulative:
                return violation
        return VIOLATION_TYPES[0]  # Fallback to first violation type
    
    def weighted_time_distribution(hours_back):
        """Generate timestamps with higher frequency during work hours"""
        now = datetime.datetime.now()
        
        # Basic random distribution across the time range
        random_hours = random.uniform(0, hours_back)
        timestamp = now - datetime.timedelta(hours=random_hours)
        
        # If we want to simulate work hours pattern
        if TIME_VARIANCE:
            # Adjust distribution to favor "work hours" (8am to 6pm)
            hour_of_day = timestamp.hour
            day_of_week = timestamp.weekday()  # 0-6 (Monday to Sunday)
            
            # If weekend (Saturday=5, Sunday=6) or outside work hours, reduce probability
            is_weekend = day_of_week >= 5
            is_work_hours = 8 <= hour_of_day <= 18
            
            if is_weekend or not is_work_hours:
                # 70% chance to regenerate the timestamp for non-work hours
                if random.random() < 0.7:
                    # Try to get a work hour time instead
                    work_day = random.randint(0, 4)  # Monday-Friday
                    work_hour = random.randint(8, 18)  # 8am-6pm
                    
                    new_timestamp = now - datetime.timedelta(
                        days=(now.weekday() - work_day) % 7,
                        hours=now.hour - work_hour,
                        minutes=now.minute,
                        seconds=now.second,
                        microseconds=now.microsecond
                    )
                    
                    # Only use the new timestamp if it's within our hours_back window
                    if (now - new_timestamp).total_seconds() <= hours_back * 3600:
                        timestamp = new_timestamp
        
        return timestamp
    
    def generate_alerts(db: Session, cameras, num_alerts: int, hours_back: int):
        """Generate mock alert data with realistic patterns"""
        logger.info(f"Generating {num_alerts} alerts...")
        
        # Delete existing alerts
        db.query(Alert).delete()
        db.commit()
        
        alerts = []
        now = datetime.datetime.now()
        
        # Get camera IDs
        camera_ids = [camera.id for camera in cameras]
        
        # Weight cameras differently - some have more alerts than others
        camera_weights = {camera.id: random.uniform(0.5, 1.5) for camera in cameras}
        total_weight = sum(camera_weights.values())
        camera_weights = {cid: weight/total_weight for cid, weight in camera_weights.items()}
        
        for i in range(num_alerts):
            # Select camera_id based on weights
            r = random.random()
            cumulative = 0
            camera_id = camera_ids[0]  # Default
            for cid, weight in camera_weights.items():
                cumulative += weight
                if r <= cumulative:
                    camera_id = cid
                    break
                    
            # Generate timestamp with work hours pattern
            created_at = weighted_time_distribution(hours_back)
            
            # Generate a violation type based on weighted probabilities
            violation_type = generate_weighted_violation()
            
            # Create alert
            alert = Alert(
                camera_id=camera_id,
                violation_type=violation_type,
                confidence=random.uniform(0.65, 0.98),
                bbox=[random.randint(50, 200), random.randint(50, 200), 
                      random.randint(250, 400), random.randint(250, 400)],
                screenshot_path=f"/screenshots/alert_{i+1}.jpg",
                created_at=created_at,
                resolved=random.random() < 0.7,  # 70% are resolved
                resolved_at=created_at + datetime.timedelta(minutes=random.randint(5, 120)) if random.random() < 0.7 else None
            )
            db.add(alert)
            
            # Commit in batches to avoid memory issues
            if (i + 1) % 50 == 0:
                db.commit()
                logger.info(f"Committed {i + 1} alerts")
        
        # Final commit
        db.commit()
        logger.info(f"Generated {num_alerts} alerts")
    
    def generate_zones(db: Session, cameras):
        """Generate mock zones for cameras"""
        logger.info("Generating zones...")
        
        # Delete existing zones
        db.query(Zone).delete()
        db.commit()
        
        # Create 1-3 zones per camera
        for camera in cameras:
            num_zones = random.randint(1, 3)
            
            for i in range(num_zones):
                # Create a random polygon (4-6 points)
                num_points = random.randint(4, 6)
                polygon = []
                
                # Generate box-like polygon with some randomness
                width, height = 1280, 720  # Assume standard HD resolution
                
                # Create a box in a random quadrant of the image
                quadrant_x = random.randint(0, 1)
                quadrant_y = random.randint(0, 1)
                
                min_x = width * quadrant_x // 2
                max_x = width * (quadrant_x + 1) // 2
                min_y = height * quadrant_y // 2
                max_y = height * (quadrant_y + 1) // 2
                
                # Generate points
                for j in range(num_points):
                    x = random.randint(min_x, max_x)
                    y = random.randint(min_y, max_y)
                    polygon.append([x, y])
                
                # Make sure it's a somewhat convex shape by sorting points by angle from center
                center_x = sum(p[0] for p in polygon) / len(polygon)
                center_y = sum(p[1] for p in polygon) / len(polygon)
                
                # Sort points by angle from center
                polygon.sort(key=lambda p: math.atan2(p[1] - center_y, p[0] - center_x))
                
                zone = Zone(
                    camera_id=camera.id,
                    name=f"Zone {i+1}",
                    polygon=polygon,
                    rule_type=random.choice(VIOLATION_TYPES)
                )
                db.add(zone)
        
        db.commit()
        logger.info("Generated zones")
    
    def main():
        """Main function to generate mock data"""
        logger.info("Starting mock data generation...")
        
        # Wait for database connection
        if not wait_for_db():
            logger.error("Database connection failed. Exiting.")
            return
            
        # Ensure database tables are created
        try:
            Base.metadata.create_all(bind=engine)
            logger.info("Database tables created successfully.")
        except Exception as e:
            logger.error(f"Error creating database tables: {e}")
            return
            
        # Create SQLAlchemy session
        db = SessionLocal()
        
        try:
            start_time = time.time()
            
            # Generate cameras
            cameras = generate_cameras(db, NUM_CAMERAS)
            
            # Generate alerts
            generate_alerts(db, cameras, NUM_ALERTS, HOURS_BACK)
            
            # Generate zones
            # generate_zones(db, cameras)  # Uncomment if you want to generate zones
            
            # Done
            end_time = time.time()
            logger.info(f"Mock data generation completed in {end_time - start_time:.2f} seconds")
            
            # Print statistics
            cameras_count = db.query(func.count(Camera.id)).scalar()
            alerts_count = db.query(func.count(Alert.id)).scalar()
            zones_count = db.query(func.count(Zone.id)).scalar()
            
            logger.info("Statistics:")
            logger.info(f"- Cameras: {cameras_count}")
            logger.info(f"- Alerts: {alerts_count}")
            logger.info(f"- Zones: {zones_count}")
            
        except Exception as e:
            logger.error(f"Error generating mock data: {e}")
            raise
        finally:
            db.close()
    
except Exception as e:
    logger.error(f"Error importing required modules: {e}")
    raise

if __name__ == "__main__":
    import math  # Import here since it's only used in generate_zones
    main() 