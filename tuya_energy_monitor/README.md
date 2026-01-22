# Tuya Energy Monitor - מוניטור אנרגיה

תוכנית לקריאת נתונים בזמן אמת ממוניטור אנרגיה עם תקשורת Tuya.

## תכונות

- 🔌 **חיבור מקומי** - תקשורת ישירה עם ההתקן ברשת המקומית
- ☁️ **חיבור ענן** - תקשורת דרך שרתי Tuya Cloud (אופציונלי)
- 📊 **קריאת נתונים בזמן אמת**:
  - מתח (Voltage)
  - זרם (Current)
  - הספק (Power)
  - צריכה מצטברת (Total Energy)
  - מקדם הספק (Power Factor) - תלוי מודל
- 💾 **שמירת נתונים** - CSV, JSON, SQLite
- 🔍 **סריקת רשת** - איתור אוטומטי של התקני Tuya

## התקנה

### דרישות מערכת
- Python 3.8 ומעלה
- רשת WiFi משותפת עם ההתקן

### התקנת תלויות

```bash
cd tuya_energy_monitor
pip install -r requirements.txt
```

## הגדרה

### שלב 1: קבלת פרטי ההתקן

כדי להתחבר להתקן Tuya, תצטרך:
1. **Device ID** - מזהה ייחודי של ההתקן
2. **IP Address** - כתובת IP ברשת המקומית
3. **Local Key** - מפתח הצפנה

#### קבלת Device ID ו-Local Key

1. היכנס ל-[Tuya IoT Platform](https://iot.tuya.com)
2. צור חשבון חדש (חינמי)
3. צור פרויקט חדש:
   - לחץ על "Create Cloud Project"
   - בחר "Smart Home" כתעשייה
   - בחר את Data Center הקרוב אליך (EU מומלץ לישראל)
4. קשר את חשבון האפליקציה:
   - עבור ל-"Link Tuya App Account"
   - סרוק QR Code מאפליקציית Tuya Smart / Smart Life
5. מצא את ההתקן:
   - עבור ל-"Devices"
   - מצא את מוניטור האנרגיה
   - העתק את Device ID ו-Local Key

#### איתור כתובת IP

הרץ את הפקודה:

```bash
python energy_monitor.py --scan
```

### שלב 2: יצירת קובץ הגדרות

העתק את קובץ הדוגמה:

```bash
cp .env.example .env
```

ערוך את `.env` עם הפרטים שלך:

```env
TUYA_DEVICE_ID=your_device_id_here
TUYA_DEVICE_IP=192.168.1.100
TUYA_LOCAL_KEY=your_local_key_here
TUYA_PROTOCOL_VERSION=3.3
```

## שימוש

### קריאת נתונים בזמן אמת

```bash
python energy_monitor.py
```

פלט לדוגמה:
```
[2024-01-15 10:30:45] מתח: 220.5V | זרם: 2.350A | הספק: 517.5W | צריכה כוללת: 123.45kWh
[2024-01-15 10:30:47] מתח: 220.3V | זרם: 2.340A | הספק: 515.5W | צריכה כוללת: 123.45kWh
```

### סריקת התקנים ברשת

```bash
python energy_monitor.py --scan
```

### שימוש עם לוגר נתונים

```python
from energy_monitor import TuyaEnergyMonitor
from data_logger import CSVLogger
from config import TuyaConfig

# Load configuration
config = TuyaConfig.from_env()

# Create monitor and logger
monitor = TuyaEnergyMonitor(config)
logger = CSVLogger("energy_data.csv")

# Add logger as callback
monitor.add_callback(logger.log)

# Connect and start monitoring
if monitor.connect():
    try:
        monitor.start_monitoring()
    finally:
        logger.close()
        monitor.disconnect()
```

### קבלת רשימת התקנים מהענן

```bash
python cloud_connector.py
```

## קובצי התוכנית

| קובץ | תיאור |
|------|--------|
| `energy_monitor.py` | מודול ראשי - חיבור וקריאת נתונים |
| `config.py` | הגדרות ופרמטרים |
| `cloud_connector.py` | חיבור ל-Tuya Cloud API |
| `data_logger.py` | שמירת נתונים לקובץ/מסד נתונים |

## מיפוי Data Points

התקני Tuya משתמשים ב-Data Points (DPs) להעברת נתונים. המיפוי הסטנדרטי:

| DP | שדה | יחידות |
|----|------|--------|
| 1 | Switch | bool |
| 17 | Total Energy | kWh × 100 |
| 18 | Current | mA |
| 19 | Power | W × 10 |
| 20 | Voltage | V × 10 |

**הערה**: חלק מההתקנים משתמשים ב-DPs שונים. אם הערכים לא נכונים, בדוק את התיעוד של ההתקן שלך או השתמש בכלי Tuya Debug.

### התאמת מיפוי DPs

```python
monitor = TuyaEnergyMonitor(config)
monitor.set_dp_mapping({
    'voltage': 106,
    'current': 104,
    'power': 105,
    'total_energy': 102,
})
```

## פתרון בעיות

### "לא ניתן להתחבר להתקן"

1. ודא שההתקן וה-PC באותה רשת WiFi
2. בדוק שכתובת ה-IP נכונה
3. ודא שה-Local Key עדכני (הוא משתנה לאחר עדכון firmware)
4. נסה גרסת פרוטוקול אחרת (3.1, 3.3, 3.4)

### "הערכים לא הגיוניים"

ייתכן שההתקן שלך משתמש במיפוי DPs שונה. נסה:
1. הדפס את הנתונים הגולמיים: `print(reading.raw_data)`
2. זהה את ה-DPs הנכונים
3. עדכן את המיפוי עם `set_dp_mapping()`

### "Local Key לא עובד"

ה-Local Key משתנה כאשר:
- ההתקן עובר reset
- ה-firmware מתעדכן
- ההתקן נמחק ונוסף מחדש לאפליקציה

קבל Local Key חדש מ-Tuya IoT Platform.

## דוגמאות נוספות

### קריאה בודדת

```python
from energy_monitor import TuyaEnergyMonitor
from config import TuyaConfig

config = TuyaConfig.from_env()
monitor = TuyaEnergyMonitor(config)

if monitor.connect():
    reading = monitor.read_once()
    if reading:
        print(f"Power: {reading.power}W")
        print(f"Energy: {reading.total_energy}kWh")
    monitor.disconnect()
```

### הפעלה/כיבוי ההתקן

```python
if monitor.connect():
    monitor.switch_on()   # הפעלה
    monitor.switch_off()  # כיבוי
```

### ניטור עם מגבלת זמן

```python
# ניטור למשך 60 שניות
monitor.start_monitoring(duration=60)

# או עד 100 קריאות
monitor.start_monitoring(max_readings=100)
```

## רישיון

MIT License

## קישורים שימושיים

- [TinyTuya Documentation](https://github.com/jasonacox/tinytuya)
- [Tuya IoT Platform](https://iot.tuya.com)
- [Tuya Smart App](https://play.google.com/store/apps/details?id=com.tuya.smart)
