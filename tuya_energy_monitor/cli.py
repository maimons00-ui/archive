#!/usr/bin/env python3
"""
Tuya Energy Monitor - Command Line Interface
ממשק שורת פקודה למוניטור אנרגיה של Tuya
"""

import argparse
import sys
import time
import logging
from datetime import datetime

from config import TuyaConfig
from energy_monitor import TuyaEnergyMonitor, discover_devices
from data_logger import create_logger, SQLiteLogger
from cloud_connector import TuyaCloudAPI, list_all_devices


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def cmd_scan(args) -> None:
    """Scan network for Tuya devices."""
    print("סורק את הרשת לאיתור התקני Tuya...")
    print("זה עשוי לקחת עד 20 שניות...\n")
    
    devices = discover_devices()
    
    if not devices:
        print("לא נמצאו התקנים.")
        print("\nטיפים:")
        print("  1. ודא שההתקן מחובר לאותה רשת WiFi")
        print("  2. נסה לכבות ולהדליק את ההתקן")
        print("  3. ודא שההתקן פועל ומגיב באפליקציה")
        return
    
    print(f"\nנמצאו {len(devices)} התקנים:\n")
    print("-" * 60)
    
    for device in devices:
        print(f"Device ID: {device.get('gwId', 'N/A')}")
        print(f"IP Address: {device.get('ip', 'N/A')}")
        print(f"Version: {device.get('version', 'N/A')}")
        print("-" * 60)


def cmd_monitor(args) -> None:
    """Start energy monitoring."""
    config = TuyaConfig.from_env()
    
    # Override with command line arguments if provided
    if args.device_id:
        config.device_id = args.device_id
    if args.ip:
        config.device_ip = args.ip
    if args.key:
        config.local_key = args.key
    if args.version:
        config.protocol_version = args.version
    if args.interval:
        config.poll_interval = args.interval
    
    # Validate configuration
    if not config.device_id or not config.device_ip or not config.local_key:
        print("שגיאה: חסרים פרטי התקן!")
        print("\nנא לספק:")
        print("  --device-id / TUYA_DEVICE_ID")
        print("  --ip / TUYA_DEVICE_IP")
        print("  --key / TUYA_LOCAL_KEY")
        print("\nאו ליצור קובץ .env")
        sys.exit(1)
    
    # Create monitor
    monitor = TuyaEnergyMonitor(config)
    
    # Setup logger if requested
    logger = None
    if args.output:
        logger = create_logger(args.format, args.output)
        monitor.add_callback(logger.log)
        print(f"שומר נתונים ל: {args.output}")
    
    # Connect
    print(f"\nמתחבר להתקן {config.device_id}...")
    
    if not monitor.connect():
        print("נכשל בהתחברות להתקן!")
        sys.exit(1)
    
    print("מחובר בהצלחה!\n")
    
    # Start monitoring
    try:
        duration = args.duration if args.duration > 0 else None
        monitor.start_monitoring(duration=duration)
    except KeyboardInterrupt:
        pass
    finally:
        if logger:
            logger.close()
        monitor.disconnect()


def cmd_read(args) -> None:
    """Read energy data once."""
    config = TuyaConfig.from_env()
    
    if args.device_id:
        config.device_id = args.device_id
    if args.ip:
        config.device_ip = args.ip
    if args.key:
        config.local_key = args.key
    
    if not config.device_id or not config.device_ip or not config.local_key:
        print("שגיאה: חסרים פרטי התקן!")
        sys.exit(1)
    
    monitor = TuyaEnergyMonitor(config)
    
    if not monitor.connect():
        print("נכשל בהתחברות!")
        sys.exit(1)
    
    reading = monitor.read_once()
    monitor.disconnect()
    
    if reading:
        if args.json:
            import json
            print(json.dumps(reading.to_dict(), indent=2, ensure_ascii=False))
        else:
            print("\n" + "=" * 50)
            print("קריאת אנרגיה")
            print("=" * 50)
            print(f"זמן:      {reading.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"מתח:      {reading.voltage:.1f} V")
            print(f"זרם:      {reading.current:.3f} A")
            print(f"הספק:     {reading.power:.1f} W")
            print(f"צריכה:    {reading.total_energy:.2f} kWh")
            if reading.switch_state is not None:
                print(f"מצב:      {'פועל' if reading.switch_state else 'כבוי'}")
            print("=" * 50)
    else:
        print("נכשל בקריאת נתונים!")
        sys.exit(1)


def cmd_control(args) -> None:
    """Control device (on/off)."""
    config = TuyaConfig.from_env()
    
    if args.device_id:
        config.device_id = args.device_id
    if args.ip:
        config.device_ip = args.ip
    if args.key:
        config.local_key = args.key
    
    if not config.device_id or not config.device_ip or not config.local_key:
        print("שגיאה: חסרים פרטי התקן!")
        sys.exit(1)
    
    monitor = TuyaEnergyMonitor(config)
    
    if not monitor.connect():
        print("נכשל בהתחברות!")
        sys.exit(1)
    
    if args.action == "on":
        success = monitor.switch_on()
    else:
        success = monitor.switch_off()
    
    monitor.disconnect()
    
    if success:
        print(f"ההתקן {'הופעל' if args.action == 'on' else 'כובה'} בהצלחה")
    else:
        print("הפעולה נכשלה!")
        sys.exit(1)


def cmd_cloud(args) -> None:
    """Cloud operations."""
    config = TuyaConfig.from_env()
    
    if not config.api_key or not config.api_secret:
        print("שגיאה: חסרים פרטי API!")
        print("\nנא להגדיר:")
        print("  TUYA_API_KEY")
        print("  TUYA_API_SECRET")
        sys.exit(1)
    
    list_all_devices(config)


def cmd_stats(args) -> None:
    """Show statistics from logged data."""
    db_path = args.database or "data/energy.db"
    
    try:
        logger = SQLiteLogger(db_path)
    except Exception as e:
        print(f"שגיאה בפתיחת מסד הנתונים: {e}")
        sys.exit(1)
    
    stats = logger.get_statistics()
    logger.close()
    
    print("\n" + "=" * 50)
    print("סטטיסטיקות אנרגיה")
    print("=" * 50)
    print(f"מספר קריאות:    {stats['reading_count']}")
    print(f"קריאה ראשונה:   {stats['first_reading']}")
    print(f"קריאה אחרונה:   {stats['last_reading']}")
    print(f"מתח ממוצע:      {stats['avg_voltage']:.1f} V" if stats['avg_voltage'] else "מתח ממוצע:      N/A")
    print(f"זרם ממוצע:      {stats['avg_current']:.3f} A" if stats['avg_current'] else "זרם ממוצע:      N/A")
    print(f"הספק ממוצע:     {stats['avg_power']:.1f} W" if stats['avg_power'] else "הספק ממוצע:     N/A")
    print(f"הספק מקסימלי:   {stats['max_power']:.1f} W" if stats['max_power'] else "הספק מקסימלי:   N/A")
    print(f"צריכה בתקופה:   {stats['energy_consumed']:.2f} kWh")
    print("=" * 50)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Tuya Energy Monitor - מוניטור אנרגיה",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
דוגמאות:
  %(prog)s scan                    # סריקת רשת לאיתור התקנים
  %(prog)s monitor                 # התחלת ניטור
  %(prog)s read                    # קריאה בודדת
  %(prog)s control on              # הפעלת ההתקן
  %(prog)s cloud                   # רשימת התקנים מהענן
        """
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='הצג מידע מפורט'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='פקודות')
    
    # Scan command
    scan_parser = subparsers.add_parser('scan', help='סריקת רשת לאיתור התקנים')
    scan_parser.set_defaults(func=cmd_scan)
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='התחלת ניטור בזמן אמת')
    monitor_parser.add_argument('--device-id', '-d', help='Device ID')
    monitor_parser.add_argument('--ip', '-i', help='כתובת IP')
    monitor_parser.add_argument('--key', '-k', help='Local Key')
    monitor_parser.add_argument('--version', '-V', type=float, default=3.3, help='גרסת פרוטוקול')
    monitor_parser.add_argument('--interval', '-n', type=float, default=2.0, help='מרווח דגימה (שניות)')
    monitor_parser.add_argument('--duration', '-t', type=float, default=0, help='משך ניטור (שניות, 0=אינסוף)')
    monitor_parser.add_argument('--output', '-o', help='תיקיית פלט לשמירת נתונים')
    monitor_parser.add_argument('--format', '-f', choices=['csv', 'json', 'sqlite', 'all'], default='csv', help='פורמט שמירה')
    monitor_parser.set_defaults(func=cmd_monitor)
    
    # Read command
    read_parser = subparsers.add_parser('read', help='קריאה בודדת')
    read_parser.add_argument('--device-id', '-d', help='Device ID')
    read_parser.add_argument('--ip', '-i', help='כתובת IP')
    read_parser.add_argument('--key', '-k', help='Local Key')
    read_parser.add_argument('--json', '-j', action='store_true', help='פלט בפורמט JSON')
    read_parser.set_defaults(func=cmd_read)
    
    # Control command
    control_parser = subparsers.add_parser('control', help='שליטה בהתקן')
    control_parser.add_argument('action', choices=['on', 'off'], help='פעולה')
    control_parser.add_argument('--device-id', '-d', help='Device ID')
    control_parser.add_argument('--ip', '-i', help='כתובת IP')
    control_parser.add_argument('--key', '-k', help='Local Key')
    control_parser.set_defaults(func=cmd_control)
    
    # Cloud command
    cloud_parser = subparsers.add_parser('cloud', help='פעולות ענן')
    cloud_parser.set_defaults(func=cmd_cloud)
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='סטטיסטיקות מנתונים שמורים')
    stats_parser.add_argument('--database', '-db', help='נתיב למסד נתונים')
    stats_parser.set_defaults(func=cmd_stats)
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    
    args.func(args)


if __name__ == "__main__":
    main()
