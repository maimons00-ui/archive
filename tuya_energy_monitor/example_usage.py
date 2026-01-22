#!/usr/bin/env python3
"""
Tuya Energy Monitor - Usage Examples
דוגמאות שימוש במוניטור אנרגיה של Tuya

הרץ את הסקריפט הזה לראות דוגמאות שונות של שימוש בספרייה.
"""

import time
from datetime import datetime


def example_basic_reading():
    """
    דוגמה 1: קריאה בסיסית של נתונים
    Basic data reading example
    """
    print("\n" + "=" * 60)
    print("דוגמה 1: קריאה בסיסית")
    print("=" * 60)
    
    from config import TuyaConfig
    from energy_monitor import TuyaEnergyMonitor
    
    # Load configuration from environment
    config = TuyaConfig.from_env()
    
    # Or create configuration manually:
    # config = TuyaConfig(
    #     device_id="your_device_id",
    #     device_ip="192.168.1.100",
    #     local_key="your_local_key",
    #     protocol_version=3.3
    # )
    
    # Create monitor instance
    monitor = TuyaEnergyMonitor(config)
    
    # Connect to device
    if monitor.connect():
        # Read data once
        reading = monitor.read_once()
        
        if reading:
            print(f"Voltage: {reading.voltage:.1f} V")
            print(f"Current: {reading.current:.3f} A")
            print(f"Power: {reading.power:.1f} W")
            print(f"Total Energy: {reading.total_energy:.2f} kWh")
        
        # Disconnect
        monitor.disconnect()
    else:
        print("Could not connect to device")


def example_continuous_monitoring():
    """
    דוגמה 2: ניטור רציף עם callback
    Continuous monitoring with callback example
    """
    print("\n" + "=" * 60)
    print("דוגמה 2: ניטור רציף")
    print("=" * 60)
    
    from config import TuyaConfig
    from energy_monitor import TuyaEnergyMonitor, EnergyReading
    
    config = TuyaConfig.from_env()
    monitor = TuyaEnergyMonitor(config)
    
    # Define callback function
    def on_reading(reading: EnergyReading):
        # This function is called for each reading
        if reading.power > 100:
            print(f"⚡ High power consumption: {reading.power:.1f}W")
        else:
            print(f"✓ Normal: {reading.power:.1f}W")
    
    # Add callback
    monitor.add_callback(on_reading)
    
    if monitor.connect():
        # Monitor for 10 seconds
        monitor.start_monitoring(duration=10)
        monitor.disconnect()


def example_with_logging():
    """
    דוגמה 3: ניטור עם שמירה לקובץ
    Monitoring with file logging example
    """
    print("\n" + "=" * 60)
    print("דוגמה 3: ניטור עם שמירה לקובץ")
    print("=" * 60)
    
    from config import TuyaConfig
    from energy_monitor import TuyaEnergyMonitor
    from data_logger import CSVLogger, SQLiteLogger, MultiLogger
    
    config = TuyaConfig.from_env()
    monitor = TuyaEnergyMonitor(config)
    
    # Create loggers
    csv_logger = CSVLogger("example_data/energy.csv")
    db_logger = SQLiteLogger("example_data/energy.db")
    
    # Use multi-logger to log to both
    multi_logger = MultiLogger([csv_logger, db_logger])
    
    # Add logger as callback
    monitor.add_callback(multi_logger.log)
    
    if monitor.connect():
        # Monitor for 30 seconds
        monitor.start_monitoring(duration=30)
        
        # Cleanup
        monitor.disconnect()
        multi_logger.close()
        
        # Print statistics
        print("\nStatistics from database:")
        db = SQLiteLogger("example_data/energy.db")
        stats = db.get_statistics()
        print(f"  Readings: {stats['reading_count']}")
        print(f"  Avg Power: {stats['avg_power']:.1f} W")
        print(f"  Energy Consumed: {stats['energy_consumed']:.3f} kWh")
        db.close()


def example_custom_dp_mapping():
    """
    דוגמה 4: התאמת מיפוי Data Points
    Custom DP mapping example
    """
    print("\n" + "=" * 60)
    print("דוגמה 4: מיפוי DP מותאם")
    print("=" * 60)
    
    from config import TuyaConfig
    from energy_monitor import TuyaEnergyMonitor
    
    config = TuyaConfig.from_env()
    monitor = TuyaEnergyMonitor(config)
    
    # Some devices use different DP IDs
    # Check raw data to find correct mapping
    monitor.set_dp_mapping({
        'voltage': 106,  # Some devices use 106 instead of 20
        'current': 104,  # Some devices use 104 instead of 18
        'power': 105,    # Some devices use 105 instead of 19
        'total_energy': 102,  # Some devices use 102 instead of 17
    })
    
    if monitor.connect():
        reading = monitor.read_once()
        
        if reading:
            print("Raw data from device:")
            print(reading.raw_data)
            print(f"\nParsed values:")
            print(f"  Voltage: {reading.voltage} V")
            print(f"  Current: {reading.current} A")
            print(f"  Power: {reading.power} W")
        
        monitor.disconnect()


def example_device_control():
    """
    דוגמה 5: שליטה בהתקן
    Device control example
    """
    print("\n" + "=" * 60)
    print("דוגמה 5: שליטה בהתקן")
    print("=" * 60)
    
    from config import TuyaConfig
    from energy_monitor import TuyaEnergyMonitor
    
    config = TuyaConfig.from_env()
    monitor = TuyaEnergyMonitor(config)
    
    if monitor.connect():
        # Check current state
        reading = monitor.read_once()
        if reading:
            print(f"Current state: {'ON' if reading.switch_state else 'OFF'}")
        
        # Turn off
        print("Turning OFF...")
        monitor.switch_off()
        time.sleep(2)
        
        # Turn on
        print("Turning ON...")
        monitor.switch_on()
        
        monitor.disconnect()


def example_energy_cost_calculation():
    """
    דוגמה 6: חישוב עלות חשמל
    Energy cost calculation example
    """
    print("\n" + "=" * 60)
    print("דוגמה 6: חישוב עלות חשמל")
    print("=" * 60)
    
    from config import TuyaConfig
    from energy_monitor import TuyaEnergyMonitor, EnergyReading
    
    # Electricity tariff in your currency per kWh
    TARIFF = 0.52  # Example: 0.52 ILS per kWh
    
    config = TuyaConfig.from_env()
    monitor = TuyaEnergyMonitor(config)
    
    start_energy = None
    
    def calculate_cost(reading: EnergyReading):
        nonlocal start_energy
        
        if start_energy is None:
            start_energy = reading.total_energy
            print(f"Starting energy: {start_energy:.2f} kWh")
            return
        
        consumed = reading.total_energy - start_energy
        cost = consumed * TARIFF
        
        print(f"Power: {reading.power:.1f}W | "
              f"Consumed: {consumed:.4f}kWh | "
              f"Cost: ₪{cost:.2f}")
    
    monitor.add_callback(calculate_cost)
    
    if monitor.connect():
        print("Monitoring energy consumption and cost...")
        print("Press Ctrl+C to stop\n")
        monitor.start_monitoring(duration=60)
        monitor.disconnect()


def example_network_scan():
    """
    דוגמה 7: סריקת רשת
    Network scanning example
    """
    print("\n" + "=" * 60)
    print("דוגמה 7: סריקת רשת")
    print("=" * 60)
    
    from energy_monitor import discover_devices
    
    print("Scanning network for Tuya devices...")
    print("This may take up to 20 seconds...\n")
    
    devices = discover_devices()
    
    if devices:
        print(f"\nFound {len(devices)} device(s):")
        for device in devices:
            print(f"  ID: {device.get('gwId')}")
            print(f"  IP: {device.get('ip')}")
            print(f"  Version: {device.get('version')}")
            print()
    else:
        print("No devices found")


def example_cloud_api():
    """
    דוגמה 8: שימוש ב-Cloud API
    Cloud API example
    """
    print("\n" + "=" * 60)
    print("דוגמה 8: Cloud API")
    print("=" * 60)
    
    from config import TuyaConfig
    from cloud_connector import TuyaCloudAPI
    
    config = TuyaConfig.from_env()
    
    if not config.api_key or not config.api_secret:
        print("Cloud API credentials not configured")
        print("Set TUYA_API_KEY and TUYA_API_SECRET in .env")
        return
    
    cloud = TuyaCloudAPI(
        api_key=config.api_key,
        api_secret=config.api_secret,
        region=config.api_region
    )
    
    # Get device list
    devices = cloud.get_device_list()
    
    for device in devices:
        print(f"Name: {device.get('name')}")
        print(f"  ID: {device.get('id')}")
        print(f"  Online: {device.get('online')}")
        
        # Get local key
        info = cloud.get_device_info(device.get('id'))
        if info:
            print(f"  Local Key: {info.get('local_key')}")
        print()


def main():
    """Run all examples with demo mode check."""
    print("""
╔════════════════════════════════════════════════════════════╗
║        Tuya Energy Monitor - דוגמאות שימוש                 ║
║                                                            ║
║  לפני הרצת הדוגמאות, ודא שיש לך קובץ .env עם:            ║
║  - TUYA_DEVICE_ID                                          ║
║  - TUYA_DEVICE_IP                                          ║
║  - TUYA_LOCAL_KEY                                          ║
╚════════════════════════════════════════════════════════════╝
    """)
    
    from config import TuyaConfig
    config = TuyaConfig.from_env()
    
    if not config.device_id or not config.device_ip or not config.local_key:
        print("⚠️  Configuration not found!")
        print("\nTo run the examples, create a .env file with your device credentials.")
        print("See .env.example for the required format.")
        print("\nRunning network scan (doesn't require configuration)...\n")
        example_network_scan()
        return
    
    print("בחר דוגמה להרצה:")
    print("1. קריאה בסיסית")
    print("2. ניטור רציף")
    print("3. ניטור עם לוגר")
    print("4. מיפוי DP מותאם")
    print("5. שליטה בהתקן")
    print("6. חישוב עלות חשמל")
    print("7. סריקת רשת")
    print("8. Cloud API")
    print("0. הרץ הכל")
    
    choice = input("\nבחירה: ").strip()
    
    examples = {
        '1': example_basic_reading,
        '2': example_continuous_monitoring,
        '3': example_with_logging,
        '4': example_custom_dp_mapping,
        '5': example_device_control,
        '6': example_energy_cost_calculation,
        '7': example_network_scan,
        '8': example_cloud_api,
    }
    
    if choice == '0':
        for func in examples.values():
            try:
                func()
            except KeyboardInterrupt:
                print("\nעצירה...")
            except Exception as e:
                print(f"Error: {e}")
    elif choice in examples:
        try:
            examples[choice]()
        except KeyboardInterrupt:
            print("\nעצירה...")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("בחירה לא חוקית")


if __name__ == "__main__":
    main()
