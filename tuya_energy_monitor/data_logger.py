"""
Tuya Energy Monitor - Data Logger
מודול לשמירת נתוני אנרגיה לקובץ

תכונות:
- שמירה לקובץ CSV
- שמירה לקובץ JSON
- תמיכה ב-SQLite
- רוטציית קבצים
"""

import os
import csv
import json
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from abc import ABC, abstractmethod

from energy_monitor import EnergyReading

logger = logging.getLogger(__name__)


class DataLogger(ABC):
    """Abstract base class for data loggers."""
    
    @abstractmethod
    def log(self, reading: EnergyReading) -> None:
        """Log a single reading."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close the logger and release resources."""
        pass


class CSVLogger(DataLogger):
    """
    CSV Data Logger
    שומר נתוני אנרגיה לקובץ CSV
    """
    
    def __init__(
        self,
        filepath: str = "energy_data.csv",
        include_raw: bool = False
    ):
        """
        Initialize CSV logger.
        
        Args:
            filepath: Path to CSV file
            include_raw: Whether to include raw data column
        """
        self.filepath = Path(filepath)
        self.include_raw = include_raw
        self._file = None
        self._writer = None
        
        # Create directory if needed
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Open file and write header if new
        file_exists = self.filepath.exists()
        self._file = open(self.filepath, 'a', newline='', encoding='utf-8')
        self._writer = csv.writer(self._file)
        
        if not file_exists:
            header = [
                'timestamp',
                'voltage_v',
                'current_a',
                'power_w',
                'total_energy_kwh',
                'power_factor',
                'frequency_hz',
                'switch_state'
            ]
            if include_raw:
                header.append('raw_data')
            self._writer.writerow(header)
            self._file.flush()
        
        logger.info(f"CSV Logger: שומר ל-{self.filepath}")
    
    def log(self, reading: EnergyReading) -> None:
        """Log reading to CSV."""
        row = [
            reading.timestamp.isoformat(),
            reading.voltage,
            reading.current,
            reading.power,
            reading.total_energy,
            reading.power_factor or '',
            reading.frequency or '',
            reading.switch_state if reading.switch_state is not None else ''
        ]
        if self.include_raw:
            row.append(json.dumps(reading.raw_data))
        
        self._writer.writerow(row)
        self._file.flush()
    
    def close(self) -> None:
        """Close CSV file."""
        if self._file:
            self._file.close()
            logger.info(f"CSV Logger סגור: {self.filepath}")


class JSONLogger(DataLogger):
    """
    JSON Data Logger
    שומר נתוני אנרגיה לקובץ JSON (שורה אחת לכל רשומה)
    """
    
    def __init__(self, filepath: str = "energy_data.jsonl"):
        """
        Initialize JSON logger (JSON Lines format).
        
        Args:
            filepath: Path to JSON Lines file
        """
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.filepath, 'a', encoding='utf-8')
        
        logger.info(f"JSON Logger: שומר ל-{self.filepath}")
    
    def log(self, reading: EnergyReading) -> None:
        """Log reading to JSON Lines file."""
        data = reading.to_dict()
        data['raw_dps'] = reading.raw_data.get('dps', {})
        
        self._file.write(json.dumps(data, ensure_ascii=False) + '\n')
        self._file.flush()
    
    def close(self) -> None:
        """Close JSON file."""
        if self._file:
            self._file.close()
            logger.info(f"JSON Logger סגור: {self.filepath}")


class SQLiteLogger(DataLogger):
    """
    SQLite Data Logger
    שומר נתוני אנרגיה למסד נתונים SQLite
    """
    
    def __init__(self, db_path: str = "energy_data.db"):
        """
        Initialize SQLite logger.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = sqlite3.connect(str(self.db_path))
        self._create_table()
        
        logger.info(f"SQLite Logger: שומר ל-{self.db_path}")
    
    def _create_table(self) -> None:
        """Create the readings table if not exists."""
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS energy_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                voltage REAL,
                current REAL,
                power REAL,
                total_energy REAL,
                power_factor REAL,
                frequency REAL,
                switch_state INTEGER,
                raw_data TEXT
            )
        ''')
        
        # Create index on timestamp
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON energy_readings(timestamp)
        ''')
        
        self.conn.commit()
    
    def log(self, reading: EnergyReading) -> None:
        """Log reading to SQLite."""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO energy_readings 
            (timestamp, voltage, current, power, total_energy, 
             power_factor, frequency, switch_state, raw_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            reading.timestamp.isoformat(),
            reading.voltage,
            reading.current,
            reading.power,
            reading.total_energy,
            reading.power_factor,
            reading.frequency,
            1 if reading.switch_state else 0 if reading.switch_state is not None else None,
            json.dumps(reading.raw_data)
        ))
        self.conn.commit()
    
    def query(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[dict]:
        """
        Query readings from database.
        
        Args:
            start_time: Start of time range
            end_time: End of time range
            limit: Maximum number of results
            
        Returns:
            List of reading dictionaries
        """
        cursor = self.conn.cursor()
        
        query = "SELECT * FROM energy_readings WHERE 1=1"
        params = []
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())
        
        query += f" ORDER BY timestamp DESC LIMIT {limit}"
        
        cursor.execute(query, params)
        columns = [desc[0] for desc in cursor.description]
        
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def get_statistics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> dict:
        """
        Get energy statistics.
        
        Args:
            start_time: Start of time range
            end_time: End of time range
            
        Returns:
            Statistics dictionary
        """
        cursor = self.conn.cursor()
        
        query = '''
            SELECT 
                COUNT(*) as reading_count,
                MIN(timestamp) as first_reading,
                MAX(timestamp) as last_reading,
                AVG(voltage) as avg_voltage,
                AVG(current) as avg_current,
                AVG(power) as avg_power,
                MAX(power) as max_power,
                MIN(total_energy) as start_energy,
                MAX(total_energy) as end_energy
            FROM energy_readings WHERE 1=1
        '''
        params = []
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())
        
        cursor.execute(query, params)
        row = cursor.fetchone()
        columns = [desc[0] for desc in cursor.description]
        stats = dict(zip(columns, row))
        
        # Calculate energy consumed
        if stats['start_energy'] and stats['end_energy']:
            stats['energy_consumed'] = stats['end_energy'] - stats['start_energy']
        else:
            stats['energy_consumed'] = 0
        
        return stats
    
    def close(self) -> None:
        """Close SQLite connection."""
        if self.conn:
            self.conn.close()
            logger.info(f"SQLite Logger סגור: {self.db_path}")


class MultiLogger(DataLogger):
    """
    Multi-destination Logger
    שומר למספר יעדים במקביל
    """
    
    def __init__(self, loggers: List[DataLogger]):
        """
        Initialize multi-logger.
        
        Args:
            loggers: List of DataLogger instances
        """
        self.loggers = loggers
    
    def log(self, reading: EnergyReading) -> None:
        """Log to all destinations."""
        for logger_instance in self.loggers:
            try:
                logger_instance.log(reading)
            except Exception as e:
                logger.error(f"Error in logger: {e}")
    
    def close(self) -> None:
        """Close all loggers."""
        for logger_instance in self.loggers:
            try:
                logger_instance.close()
            except Exception as e:
                logger.error(f"Error closing logger: {e}")


def create_logger(
    output_format: str = "csv",
    output_path: Optional[str] = None
) -> DataLogger:
    """
    Factory function to create appropriate logger.
    
    Args:
        output_format: "csv", "json", "sqlite", or "all"
        output_path: Base path for output files
        
    Returns:
        DataLogger instance
    """
    base_path = output_path or "data"
    timestamp = datetime.now().strftime("%Y%m%d")
    
    if output_format == "csv":
        return CSVLogger(f"{base_path}/energy_{timestamp}.csv")
    elif output_format == "json":
        return JSONLogger(f"{base_path}/energy_{timestamp}.jsonl")
    elif output_format == "sqlite":
        return SQLiteLogger(f"{base_path}/energy.db")
    elif output_format == "all":
        return MultiLogger([
            CSVLogger(f"{base_path}/energy_{timestamp}.csv"),
            JSONLogger(f"{base_path}/energy_{timestamp}.jsonl"),
            SQLiteLogger(f"{base_path}/energy.db"),
        ])
    else:
        raise ValueError(f"Unknown format: {output_format}")


if __name__ == "__main__":
    # Test loggers
    from datetime import datetime
    
    # Create test reading
    test_reading = EnergyReading(
        timestamp=datetime.now(),
        voltage=220.5,
        current=2.35,
        power=518.0,
        total_energy=123.45,
        switch_state=True,
        raw_data={"dps": {"1": True, "18": 2350, "19": 5180, "20": 2205}}
    )
    
    # Test each logger
    for fmt in ["csv", "json", "sqlite"]:
        print(f"Testing {fmt} logger...")
        log = create_logger(fmt, "test_data")
        log.log(test_reading)
        log.close()
        print(f"  ✓ {fmt} logger works")
    
    # Test SQLite query
    db_logger = SQLiteLogger("test_data/energy.db")
    stats = db_logger.get_statistics()
    print(f"\nDatabase statistics: {stats}")
    db_logger.close()
