"""
Tuya Energy Monitor - Configuration Module
קובץ הגדרות לחיבור למוניטור אנרגיה של Tuya
"""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class TuyaConfig:
    """Configuration class for Tuya device connection."""
    
    # Device credentials (for local connection)
    device_id: str
    device_ip: str
    local_key: str
    
    # Cloud API credentials (optional, for cloud connection)
    api_region: str = "eu"  # Options: cn, eu, us, in
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    
    # Connection settings
    protocol_version: float = 3.3  # Common versions: 3.1, 3.3, 3.4
    connection_timeout: int = 10
    retry_count: int = 3
    
    # Polling settings
    poll_interval: float = 2.0  # Seconds between readings
    
    @classmethod
    def from_env(cls) -> "TuyaConfig":
        """Create configuration from environment variables."""
        return cls(
            device_id=os.getenv("TUYA_DEVICE_ID", ""),
            device_ip=os.getenv("TUYA_DEVICE_IP", ""),
            local_key=os.getenv("TUYA_LOCAL_KEY", ""),
            api_region=os.getenv("TUYA_API_REGION", "eu"),
            api_key=os.getenv("TUYA_API_KEY"),
            api_secret=os.getenv("TUYA_API_SECRET"),
            protocol_version=float(os.getenv("TUYA_PROTOCOL_VERSION", "3.3")),
            connection_timeout=int(os.getenv("TUYA_TIMEOUT", "10")),
            retry_count=int(os.getenv("TUYA_RETRY_COUNT", "3")),
            poll_interval=float(os.getenv("TUYA_POLL_INTERVAL", "2.0")),
        )


# Energy Monitor Data Point (DP) IDs - Common mappings for energy monitors
# These may vary depending on the specific device model
class EnergyDataPoints:
    """Common Data Point IDs for Tuya energy monitors."""
    
    # Basic switch
    SWITCH = 1
    
    # Power measurements
    CURRENT = 18          # Current in mA (milliamperes)
    POWER = 19            # Power in W/10 (divide by 10 for watts)
    VOLTAGE = 20          # Voltage in V/10 (divide by 10 for volts)
    
    # Energy consumption
    TOTAL_ENERGY = 17     # Total energy in kWh * 100
    
    # Additional DPs that some devices may have
    POWER_FACTOR = 21     # Power factor * 100
    FREQUENCY = 22        # Frequency in Hz * 100
    
    # Alternative DP mappings for some device models
    ALT_CURRENT = 104
    ALT_POWER = 105
    ALT_VOLTAGE = 106
    ALT_TOTAL_ENERGY = 102


# Example .env file content
ENV_TEMPLATE = """
# Tuya Device Configuration
# פרטי התקן Tuya

# Device ID - מזהה ההתקן (ניתן למצוא באפליקציית Tuya Smart או Smart Life)
TUYA_DEVICE_ID=your_device_id_here

# Device IP Address - כתובת IP של ההתקן ברשת המקומית
TUYA_DEVICE_IP=192.168.1.100

# Local Key - מפתח מקומי (ניתן לקבל מ-Tuya IoT Platform)
TUYA_LOCAL_KEY=your_local_key_here

# Protocol Version - גרסת פרוטוקול (בדרך כלל 3.3 או 3.4)
TUYA_PROTOCOL_VERSION=3.3

# Tuya Cloud API Credentials (אופציונלי - לחיבור דרך הענן)
TUYA_API_REGION=eu
TUYA_API_KEY=your_api_key_here
TUYA_API_SECRET=your_api_secret_here

# Connection Settings
TUYA_TIMEOUT=10
TUYA_RETRY_COUNT=3
TUYA_POLL_INTERVAL=2.0
"""
