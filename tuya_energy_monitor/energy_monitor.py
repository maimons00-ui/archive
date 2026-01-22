"""
Tuya Energy Monitor - Main Module
מודול ראשי לקריאת נתונים ממוניטור אנרגיה של Tuya בזמן אמת

Author: AI Assistant
Description: Real-time energy monitoring from Tuya smart energy meters
"""

import time
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field

import tinytuya

from config import TuyaConfig, EnergyDataPoints

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class EnergyReading:
    """Data class for energy monitor readings."""
    
    timestamp: datetime
    voltage: float = 0.0           # Volts (V)
    current: float = 0.0           # Amperes (A)
    power: float = 0.0             # Watts (W)
    total_energy: float = 0.0      # Kilowatt-hours (kWh)
    power_factor: Optional[float] = None
    frequency: Optional[float] = None
    switch_state: Optional[bool] = None
    raw_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert reading to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "voltage_v": self.voltage,
            "current_a": self.current,
            "power_w": self.power,
            "total_energy_kwh": self.total_energy,
            "power_factor": self.power_factor,
            "frequency_hz": self.frequency,
            "switch_state": self.switch_state,
        }
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"[{self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}] "
            f"מתח: {self.voltage:.1f}V | "
            f"זרם: {self.current:.3f}A | "
            f"הספק: {self.power:.1f}W | "
            f"צריכה כוללת: {self.total_energy:.2f}kWh"
        )


class TuyaEnergyMonitor:
    """
    Tuya Energy Monitor Client
    לקוח לקריאת נתונים ממוניטור אנרגיה של Tuya
    
    Supports both local and cloud connections.
    """
    
    def __init__(self, config: TuyaConfig):
        """
        Initialize the energy monitor client.
        
        Args:
            config: TuyaConfig object with device credentials
        """
        self.config = config
        self.device: Optional[tinytuya.OutletDevice] = None
        self._connected = False
        self._callbacks: list[Callable[[EnergyReading], None]] = []
        
        # Data point mapping (can be customized per device)
        self.dp_mapping = {
            'voltage': EnergyDataPoints.VOLTAGE,
            'current': EnergyDataPoints.CURRENT,
            'power': EnergyDataPoints.POWER,
            'total_energy': EnergyDataPoints.TOTAL_ENERGY,
            'switch': EnergyDataPoints.SWITCH,
            'power_factor': EnergyDataPoints.POWER_FACTOR,
            'frequency': EnergyDataPoints.FREQUENCY,
        }
    
    def set_dp_mapping(self, mapping: Dict[str, int]) -> None:
        """
        Set custom data point mapping for your specific device.
        
        Args:
            mapping: Dictionary mapping field names to DP IDs
        """
        self.dp_mapping.update(mapping)
        logger.info(f"Updated DP mapping: {self.dp_mapping}")
    
    def connect(self) -> bool:
        """
        Establish connection to the Tuya device.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            logger.info(f"מתחבר להתקן Tuya: {self.config.device_id}")
            logger.info(f"כתובת IP: {self.config.device_ip}")
            
            # Create device instance
            self.device = tinytuya.OutletDevice(
                dev_id=self.config.device_id,
                address=self.config.device_ip,
                local_key=self.config.local_key,
                version=self.config.protocol_version
            )
            
            # Set connection timeout
            self.device.set_socketTimeout(self.config.connection_timeout)
            
            # Test connection by getting status
            status = self.device.status()
            
            if 'Error' in status:
                logger.error(f"שגיאת חיבור: {status['Error']}")
                self._connected = False
                return False
            
            logger.info("חיבור הצליח! ✓")
            logger.debug(f"Device status: {status}")
            self._connected = True
            return True
            
        except Exception as e:
            logger.error(f"שגיאה בחיבור להתקן: {e}")
            self._connected = False
            return False
    
    def disconnect(self) -> None:
        """Disconnect from the device."""
        if self.device:
            try:
                self.device.close()
            except:
                pass
        self._connected = False
        logger.info("התנתק מההתקן")
    
    def is_connected(self) -> bool:
        """Check if connected to device."""
        return self._connected
    
    def _parse_reading(self, raw_data: Dict[str, Any]) -> EnergyReading:
        """
        Parse raw device data into EnergyReading object.
        
        Args:
            raw_data: Raw data from device
            
        Returns:
            EnergyReading object with parsed values
        """
        dps = raw_data.get('dps', {})
        
        # Get values from data points with appropriate conversions
        # Note: Conversion factors may vary by device model
        
        voltage = 0.0
        voltage_dp = str(self.dp_mapping['voltage'])
        if voltage_dp in dps:
            voltage = dps[voltage_dp] / 10.0  # Usually V * 10
        
        current = 0.0
        current_dp = str(self.dp_mapping['current'])
        if current_dp in dps:
            current = dps[current_dp] / 1000.0  # Usually mA
        
        power = 0.0
        power_dp = str(self.dp_mapping['power'])
        if power_dp in dps:
            power = dps[power_dp] / 10.0  # Usually W * 10
        
        total_energy = 0.0
        energy_dp = str(self.dp_mapping['total_energy'])
        if energy_dp in dps:
            total_energy = dps[energy_dp] / 100.0  # Usually kWh * 100
        
        switch_state = None
        switch_dp = str(self.dp_mapping['switch'])
        if switch_dp in dps:
            switch_state = bool(dps[switch_dp])
        
        power_factor = None
        pf_dp = str(self.dp_mapping.get('power_factor', 0))
        if pf_dp in dps:
            power_factor = dps[pf_dp] / 100.0
        
        frequency = None
        freq_dp = str(self.dp_mapping.get('frequency', 0))
        if freq_dp in dps:
            frequency = dps[freq_dp] / 100.0
        
        return EnergyReading(
            timestamp=datetime.now(),
            voltage=voltage,
            current=current,
            power=power,
            total_energy=total_energy,
            power_factor=power_factor,
            frequency=frequency,
            switch_state=switch_state,
            raw_data=raw_data
        )
    
    def read_once(self) -> Optional[EnergyReading]:
        """
        Read current energy data once.
        
        Returns:
            EnergyReading object or None if failed
        """
        if not self._connected or not self.device:
            logger.warning("לא מחובר להתקן. נא להתחבר תחילה.")
            return None
        
        try:
            # Get device status
            status = self.device.status()
            
            if 'Error' in status:
                logger.error(f"שגיאה בקריאת נתונים: {status['Error']}")
                return None
            
            reading = self._parse_reading(status)
            
            # Trigger callbacks
            for callback in self._callbacks:
                try:
                    callback(reading)
                except Exception as e:
                    logger.error(f"Error in callback: {e}")
            
            return reading
            
        except Exception as e:
            logger.error(f"שגיאה בקריאת נתונים: {e}")
            return None
    
    def add_callback(self, callback: Callable[[EnergyReading], None]) -> None:
        """
        Add a callback function to be called on each reading.
        
        Args:
            callback: Function that takes EnergyReading as argument
        """
        self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[EnergyReading], None]) -> None:
        """Remove a callback function."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def start_monitoring(
        self,
        duration: Optional[float] = None,
        max_readings: Optional[int] = None
    ) -> None:
        """
        Start continuous monitoring of energy data.
        
        Args:
            duration: Maximum duration in seconds (None for infinite)
            max_readings: Maximum number of readings (None for infinite)
        """
        if not self._connected:
            logger.error("לא מחובר להתקן!")
            return
        
        logger.info("=" * 60)
        logger.info("מתחיל ניטור אנרגיה בזמן אמת")
        logger.info(f"מרווח דגימה: {self.config.poll_interval} שניות")
        logger.info("לחץ Ctrl+C לעצירה")
        logger.info("=" * 60)
        
        start_time = time.time()
        reading_count = 0
        
        try:
            while True:
                reading = self.read_once()
                
                if reading:
                    reading_count += 1
                    print(reading)
                
                # Check stop conditions
                if duration and (time.time() - start_time) >= duration:
                    logger.info(f"הגיע לזמן מקסימלי ({duration}s)")
                    break
                
                if max_readings and reading_count >= max_readings:
                    logger.info(f"הגיע למספר קריאות מקסימלי ({max_readings})")
                    break
                
                time.sleep(self.config.poll_interval)
                
        except KeyboardInterrupt:
            logger.info("\nניטור הופסק על ידי המשתמש")
        
        logger.info(f"סה״כ קריאות: {reading_count}")
    
    def get_device_info(self) -> Optional[Dict[str, Any]]:
        """
        Get device information.
        
        Returns:
            Device info dictionary or None
        """
        if not self.device:
            return None
        
        return {
            "device_id": self.config.device_id,
            "ip_address": self.config.device_ip,
            "protocol_version": self.config.protocol_version,
            "connected": self._connected,
        }
    
    def switch_on(self) -> bool:
        """Turn on the device switch."""
        if not self.device:
            return False
        try:
            self.device.turn_on()
            logger.info("ההתקן הופעל")
            return True
        except Exception as e:
            logger.error(f"שגיאה בהפעלת ההתקן: {e}")
            return False
    
    def switch_off(self) -> bool:
        """Turn off the device switch."""
        if not self.device:
            return False
        try:
            self.device.turn_off()
            logger.info("ההתקן כובה")
            return True
        except Exception as e:
            logger.error(f"שגיאה בכיבוי ההתקן: {e}")
            return False


def discover_devices() -> list[Dict[str, Any]]:
    """
    Scan network for Tuya devices.
    סריקת רשת לאיתור התקני Tuya
    
    Returns:
        List of discovered devices
    """
    logger.info("סורק את הרשת לאיתור התקני Tuya...")
    logger.info("זה עשוי לקחת מספר שניות...")
    
    devices = tinytuya.deviceScan(verbose=False)
    
    if devices:
        logger.info(f"נמצאו {len(devices)} התקנים:")
        for ip, device in devices.items():
            logger.info(f"  - IP: {ip}, ID: {device.get('gwId', 'N/A')}")
    else:
        logger.info("לא נמצאו התקנים")
    
    return list(devices.values()) if devices else []


def main():
    """Main entry point for the energy monitor."""
    print("=" * 60)
    print("  Tuya Energy Monitor - מוניטור אנרגיה")
    print("  קריאת נתונים בזמן אמת ממוניטור אנרגיה של Tuya")
    print("=" * 60)
    print()
    
    # Load configuration from environment
    config = TuyaConfig.from_env()
    
    # Validate configuration
    if not config.device_id or not config.device_ip or not config.local_key:
        print("שגיאה: חסרים פרטי התקן!")
        print()
        print("נא ליצור קובץ .env עם הפרטים הבאים:")
        print("-" * 40)
        print("TUYA_DEVICE_ID=your_device_id")
        print("TUYA_DEVICE_IP=192.168.1.xxx")
        print("TUYA_LOCAL_KEY=your_local_key")
        print("-" * 40)
        print()
        print("לאיתור התקנים ברשת, הרץ:")
        print("  python energy_monitor.py --scan")
        print()
        print("לקבלת פרטי ההתקן (Device ID ו-Local Key):")
        print("  1. היכנס ל-https://iot.tuya.com")
        print("  2. צור פרויקט חדש")
        print("  3. קשר את ההתקן דרך האפליקציה")
        print("  4. מצא את הפרטים ב-Device Management")
        return
    
    # Create monitor instance
    monitor = TuyaEnergyMonitor(config)
    
    # Example: Custom DP mapping for specific device
    # Uncomment and modify if your device uses different DPs
    # monitor.set_dp_mapping({
    #     'voltage': 106,
    #     'current': 104,
    #     'power': 105,
    #     'total_energy': 102,
    # })
    
    # Connect to device
    if not monitor.connect():
        print("נכשל בהתחברות להתקן")
        return
    
    try:
        # Start continuous monitoring
        monitor.start_monitoring()
    finally:
        # Cleanup
        monitor.disconnect()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--scan":
        # Scan mode
        discover_devices()
    else:
        # Normal monitoring mode
        main()
