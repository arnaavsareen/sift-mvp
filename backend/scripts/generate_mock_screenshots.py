#!/usr/bin/env python3
"""
Script to generate mock screenshot images for alerts.
This will create placeholder images in the screenshots directory.
"""

import os
import sys
import random
import logging
from pathlib import Path
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("mock_screenshots_generator")

# Add parent directory to path to import backend modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    
    from backend.database import SessionLocal, engine
    from backend.models import Alert
    from backend.config import SCREENSHOTS_DIR
    
    # Create screenshots directory if it doesn't exist
    os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
    
    def generate_mock_screenshot(alert_id, violation_type, size=(640, 480)):
        """
        Generate a mock screenshot for an alert with a colored box indicating the violation
        """
        # Create a black background image
        img = Image.new('RGB', size, color=(0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Add some random background elements to make it look like a real scene
        # Draw random lines
        for _ in range(30):
            x1, y1 = random.randint(0, size[0]), random.randint(0, size[1])
            x2, y2 = random.randint(0, size[0]), random.randint(0, size[1])
            color = (random.randint(30, 100), random.randint(30, 100), random.randint(30, 100))
            draw.line([(x1, y1), (x2, y2)], fill=color, width=random.randint(1, 3))
        
        # Draw some random rectangles in the background
        for _ in range(10):
            x1, y1 = random.randint(0, size[0]), random.randint(0, size[1])
            x2, y2 = x1 + random.randint(50, 200), y1 + random.randint(50, 150)
            color = (random.randint(20, 80), random.randint(20, 80), random.randint(20, 80))
            draw.rectangle([x1, y1, x2, y2], outline=color, width=1)
        
        # Generate a person silhouette
        person_width = random.randint(80, 120)
        person_height = random.randint(150, 250)
        person_x = random.randint(50, size[0] - person_width - 50)
        person_y = random.randint(50, size[1] - person_height - 50)
        
        # Draw person silhouette (simplified)
        person_color = (random.randint(100, 200), random.randint(100, 200), random.randint(100, 200))
        
        # Body
        draw.rectangle([person_x, person_y + person_height // 4, 
                       person_x + person_width, person_y + person_height],
                      fill=person_color)
        
        # Head
        head_size = person_width * 0.7
        draw.ellipse([person_x + (person_width - head_size) // 2, 
                      person_y,
                      person_x + (person_width + head_size) // 2, 
                      person_y + person_height // 4],
                     fill=person_color)
        
        # Draw detection box
        # Box around the person with violation type color
        box_color = {
            "no_hardhat": (255, 50, 50),      # Red
            "no_safety_vest": (255, 153, 51),  # Orange
            "unsafe_distance": (255, 255, 51), # Yellow
            "restricted_area": (153, 51, 255), # Purple
            "improper_lifting": (51, 153, 255), # Blue
            "no_safety_goggles": (51, 255, 153) # Green
        }.get(violation_type, (255, 50, 50))
        
        # Make the box slightly larger than the person
        box_padding = 10
        draw.rectangle([person_x - box_padding, 
                       person_y - box_padding, 
                       person_x + person_width + box_padding, 
                       person_y + person_height + box_padding],
                      outline=box_color, width=3)
        
        # Add text for violation
        violation_text = {
            "no_hardhat": "No Hardhat",
            "no_safety_vest": "No Safety Vest",
            "unsafe_distance": "Unsafe Distance",
            "restricted_area": "Restricted Area",
            "improper_lifting": "Improper Lifting",
            "no_safety_goggles": "No Safety Goggles"
        }.get(violation_type, violation_type)
        
        # Add violation text with colored background
        text_width = len(violation_text) * 8  # Approx width
        text_height = 20
        text_x = person_x - box_padding
        text_y = person_y - box_padding - text_height
        
        # Draw text background
        draw.rectangle([text_x, text_y, text_x + text_width, text_y + text_height],
                      fill=box_color)
        
        # Try to use a font, fallback to default if not available
        try:
            font = ImageFont.truetype("Arial", 14)
        except IOError:
            try:
                # Try system fonts that might be available in Docker
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
            except:
                font = ImageFont.load_default()
        
        # Draw text
        draw.text((text_x + 5, text_y + 2), violation_text, fill=(255, 255, 255), font=font)
        
        # Add timestamp
        timestamp = "2023-06-15 14:32:45"
        try:
            draw.text((10, size[1] - 25), timestamp, fill=(200, 200, 200), font=font)
        except:
            draw.text((10, size[1] - 25), timestamp, fill=(200, 200, 200))
        
        # Add alert ID
        try:
            draw.text((size[0] - 100, size[1] - 25), f"Alert #{alert_id}", fill=(200, 200, 200), font=font)
        except:
            draw.text((size[0] - 100, size[1] - 25), f"Alert #{alert_id}", fill=(200, 200, 200))
        
        return img
    
    def wait_for_alerts(max_retries=30, retry_interval=2):
        """Wait for alerts to be available in the database"""
        logger.info("Checking for alerts in the database...")
        db = SessionLocal()
        
        retries = 0
        while retries < max_retries:
            try:
                # Try to get alert count
                alert_count = db.query(Alert).count()
                
                if alert_count > 0:
                    logger.info(f"Found {alert_count} alerts in the database!")
                    db.close()
                    return True
                
                retries += 1
                if retries >= max_retries:
                    logger.warning(f"No alerts found in database after {max_retries} attempts.")
                    db.close()
                    return False
                
                logger.warning(f"No alerts found yet. Retrying in {retry_interval}s... ({retries}/{max_retries})")
                time.sleep(retry_interval)
                
            except Exception as e:
                retries += 1
                if retries >= max_retries:
                    logger.error(f"Failed to check alerts after {max_retries} attempts: {e}")
                    db.close()
                    return False
                logger.warning(f"Database error: {e}. Retrying in {retry_interval}s... ({retries}/{max_retries})")
                time.sleep(retry_interval)
        
        db.close()
        return False
    
    def main():
        """Generate mock screenshots for all existing alerts"""
        logger.info("Starting mock screenshot generation...")
        
        # Wait for alerts to be available
        if not wait_for_alerts():
            logger.warning("No alerts found to generate screenshots for. Exiting.")
            return
            
        # Create SQLAlchemy session
        db = SessionLocal()
        
        try:
            # Get all alerts
            alerts = db.query(Alert).all()
            logger.info(f"Generating {len(alerts)} mock screenshots...")
            
            # Create screenshots directory if it doesn't exist
            os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
            logger.info(f"Screenshots will be saved to: {SCREENSHOTS_DIR}")
            
            for i, alert in enumerate(alerts):
                # Generate mock image path
                screenshot_path = f"/screenshots/alert_{alert.id}.jpg"
                full_path = os.path.join(SCREENSHOTS_DIR, f"alert_{alert.id}.jpg")
                
                # Generate mock image
                img = generate_mock_screenshot(alert.id, alert.violation_type)
                
                # Save the image
                img.save(full_path)
                
                # Update alert with screenshot path
                alert.screenshot_path = screenshot_path
                
                # Print progress every 10 images
                if (i + 1) % 10 == 0:
                    logger.info(f"Generated {i + 1}/{len(alerts)} screenshots")
            
            # Commit updates
            db.commit()
            logger.info(f"Generated {len(alerts)} screenshots")
            
        except Exception as e:
            logger.error(f"Error generating screenshots: {e}")
            raise
        finally:
            db.close()
    
except ImportError as e:
    logger.error(f"Error importing required modules: {e}")
    logger.error("Make sure PIL and other dependencies are installed: pip install Pillow numpy")
    raise
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    raise

if __name__ == "__main__":
    main() 