"""
Tuya Cloud API Connector
מחבר ל-API של Tuya Cloud לקבלת מידע על התקנים ונתונים היסטוריים

זה מודול אופציונלי לחיבור דרך הענן במקום חיבור מקומי.
"""

import time
import hmac
import hashlib
import json
import logging
from typing import Dict, Any, Optional, List
from urllib.parse import urlencode

import requests

from config import TuyaConfig

logger = logging.getLogger(__name__)


class TuyaCloudAPI:
    """
    Tuya Cloud API Client
    לקוח API לענן של Tuya
    
    שימושים:
    - קבלת רשימת התקנים
    - קבלת Local Key של התקנים
    - קריאת נתונים היסטוריים
    - שליטה מרחוק בהתקנים
    """
    
    # API endpoints by region
    ENDPOINTS = {
        "cn": "https://openapi.tuyacn.com",
        "eu": "https://openapi.tuyaeu.com",
        "us": "https://openapi.tuyaus.com",
        "in": "https://openapi.tuyain.com",
    }
    
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        region: str = "eu"
    ):
        """
        Initialize Tuya Cloud API client.
        
        Args:
            api_key: Tuya IoT Platform Access ID
            api_secret: Tuya IoT Platform Access Secret
            region: API region (cn, eu, us, in)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.endpoint = self.ENDPOINTS.get(region, self.ENDPOINTS["eu"])
        self.access_token: Optional[str] = None
        self.token_expire_time: int = 0
    
    def _sign(
        self,
        method: str,
        path: str,
        params: Optional[Dict] = None,
        body: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Generate signature for API request.
        
        Args:
            method: HTTP method
            path: API path
            params: Query parameters
            body: Request body
            
        Returns:
            Headers with signature
        """
        t = str(int(time.time() * 1000))
        
        # Content hash
        content_hash = hashlib.sha256(
            (body or "").encode()
        ).hexdigest()
        
        # String to sign
        str_to_sign = f"{method}\n{content_hash}\n\n{path}"
        
        if self.access_token:
            sign_str = f"{self.api_key}{self.access_token}{t}{str_to_sign}"
        else:
            sign_str = f"{self.api_key}{t}{str_to_sign}"
        
        # Calculate signature
        sign = hmac.new(
            self.api_secret.encode(),
            sign_str.encode(),
            hashlib.sha256
        ).hexdigest().upper()
        
        headers = {
            "client_id": self.api_key,
            "sign": sign,
            "t": t,
            "sign_method": "HMAC-SHA256",
            "Content-Type": "application/json",
        }
        
        if self.access_token:
            headers["access_token"] = self.access_token
        
        return headers
    
    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict] = None,
        body: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Make API request.
        
        Args:
            method: HTTP method
            path: API path
            params: Query parameters
            body: Request body
            
        Returns:
            API response
        """
        url = f"{self.endpoint}{path}"
        body_str = json.dumps(body) if body else None
        headers = self._sign(method, path, params, body_str)
        
        if params:
            url = f"{url}?{urlencode(params)}"
        
        response = requests.request(
            method=method,
            url=url,
            headers=headers,
            data=body_str
        )
        
        return response.json()
    
    def get_token(self) -> bool:
        """
        Get access token from Tuya Cloud.
        
        Returns:
            True if successful
        """
        result = self._request("GET", "/v1.0/token?grant_type=1")
        
        if result.get("success"):
            data = result["result"]
            self.access_token = data["access_token"]
            self.token_expire_time = time.time() + data["expire_time"]
            logger.info("קיבלתי token מ-Tuya Cloud")
            return True
        else:
            logger.error(f"שגיאה בקבלת token: {result.get('msg')}")
            return False
    
    def ensure_token(self) -> bool:
        """Ensure we have a valid token."""
        if not self.access_token or time.time() >= self.token_expire_time - 60:
            return self.get_token()
        return True
    
    def get_device_list(self) -> List[Dict[str, Any]]:
        """
        Get list of all devices.
        
        Returns:
            List of device dictionaries
        """
        if not self.ensure_token():
            return []
        
        result = self._request("GET", "/v1.0/iot-01/associated-users/devices")
        
        if result.get("success"):
            devices = result.get("result", {}).get("devices", [])
            logger.info(f"נמצאו {len(devices)} התקנים")
            return devices
        else:
            logger.error(f"שגיאה בקבלת רשימת התקנים: {result.get('msg')}")
            return []
    
    def get_device_info(self, device_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed device information including local_key.
        
        Args:
            device_id: Device ID
            
        Returns:
            Device information dictionary
        """
        if not self.ensure_token():
            return None
        
        result = self._request("GET", f"/v1.0/devices/{device_id}")
        
        if result.get("success"):
            return result.get("result")
        else:
            logger.error(f"שגיאה בקבלת מידע על התקן: {result.get('msg')}")
            return None
    
    def get_device_status(self, device_id: str) -> Optional[List[Dict[str, Any]]]:
        """
        Get current device status (data points).
        
        Args:
            device_id: Device ID
            
        Returns:
            List of status dictionaries
        """
        if not self.ensure_token():
            return None
        
        result = self._request("GET", f"/v1.0/devices/{device_id}/status")
        
        if result.get("success"):
            return result.get("result", [])
        else:
            logger.error(f"שגיאה בקבלת סטטוס התקן: {result.get('msg')}")
            return None
    
    def send_command(
        self,
        device_id: str,
        commands: List[Dict[str, Any]]
    ) -> bool:
        """
        Send commands to device.
        
        Args:
            device_id: Device ID
            commands: List of command dictionaries
            
        Returns:
            True if successful
        """
        if not self.ensure_token():
            return False
        
        body = {"commands": commands}
        result = self._request(
            "POST",
            f"/v1.0/devices/{device_id}/commands",
            body=body
        )
        
        if result.get("success"):
            logger.info("פקודה נשלחה בהצלחה")
            return True
        else:
            logger.error(f"שגיאה בשליחת פקודה: {result.get('msg')}")
            return False
    
    def get_device_logs(
        self,
        device_id: str,
        start_time: int,
        end_time: int,
        type_: str = "7"
    ) -> List[Dict[str, Any]]:
        """
        Get device logs/history.
        
        Args:
            device_id: Device ID
            start_time: Start timestamp (ms)
            end_time: End timestamp (ms)
            type_: Log type (7 = status reports)
            
        Returns:
            List of log entries
        """
        if not self.ensure_token():
            return []
        
        params = {
            "start_time": start_time,
            "end_time": end_time,
            "type": type_,
            "size": 100
        }
        
        result = self._request(
            "GET",
            f"/v1.0/devices/{device_id}/logs",
            params=params
        )
        
        if result.get("success"):
            return result.get("result", {}).get("logs", [])
        else:
            logger.error(f"שגיאה בקבלת לוגים: {result.get('msg')}")
            return []
    
    def get_local_key(self, device_id: str) -> Optional[str]:
        """
        Get the local_key for a device.
        
        Args:
            device_id: Device ID
            
        Returns:
            Local key string or None
        """
        info = self.get_device_info(device_id)
        if info:
            return info.get("local_key")
        return None


def list_all_devices(config: TuyaConfig) -> None:
    """
    Utility function to list all devices and their local keys.
    
    Args:
        config: TuyaConfig with cloud API credentials
    """
    if not config.api_key or not config.api_secret:
        print("שגיאה: חסרים פרטי API")
        print("נא להגדיר TUYA_API_KEY ו-TUYA_API_SECRET")
        return
    
    cloud = TuyaCloudAPI(
        api_key=config.api_key,
        api_secret=config.api_secret,
        region=config.api_region
    )
    
    devices = cloud.get_device_list()
    
    if not devices:
        print("לא נמצאו התקנים")
        return
    
    print("\n" + "=" * 80)
    print("רשימת התקני Tuya:")
    print("=" * 80)
    
    for device in devices:
        device_id = device.get("id", "N/A")
        name = device.get("name", "Unknown")
        category = device.get("category", "N/A")
        online = "מחובר" if device.get("online") else "מנותק"
        
        # Get local key
        info = cloud.get_device_info(device_id)
        local_key = info.get("local_key", "N/A") if info else "N/A"
        
        print(f"\nשם: {name}")
        print(f"  Device ID: {device_id}")
        print(f"  Local Key: {local_key}")
        print(f"  קטגוריה: {category}")
        print(f"  סטטוס: {online}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Test cloud connection
    config = TuyaConfig.from_env()
    list_all_devices(config)
