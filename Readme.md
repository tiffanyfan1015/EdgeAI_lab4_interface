# Edge AI Lab4 Web Interface
## Environment
Python 3.8+, Flask + SQLite
```
pip install flask flask_sqlalchemy
```
## Step
1. Run the server
```
python app.py
```
2. Find IP of your device
3. Use cellphone(or any browser), visit `http://<Your device's IP>:8000`

## Structure
```
LAB4_interface/
├── app.py                  # Main Flask application
├── instance/
│   └── database.db         # SQLite database
├── static/
│   └── uploads/            # Where uploads pictures stored
├── templates/              # All HTML pages
│   ├── index.html          # Home page menu (links to subpages)
│   ├── data.html           # Upload people's name, description, photo
│   ├── active.html         # Who's there? (可以接模型輸出結果)
│   └── control.html        # 手動設某人為 active
└── Readme.md               

```

## Usage
| 頁面           | 位置         | 功能                              |
| ------------ | ---------- | ------------------------------- |
| 🏠 首頁        | `/`        | 顯示三個功能入口（data, active, control） |
| 📤 上傳資料      | `/data`    | 上傳圖片、敘述，並即時顯示人物列表               |
| 🧠 Active 狀態 | `/active`  | 顯示目前被辨識為「靠近者」的人物與語言模型回應         |
| 🛠️ 手動控制台(測試用)    | `/control` | 點選人物按鈕，手動將其設為 active 狀態         |

## Some API
| API              | 方法   | 說明                     |
| ---------------- | ---- | ---------------------- |
| `/upload`        | POST | 上傳圖片與敘述（由表單觸發）         |
| `/list`          | GET  | 回傳目前所有人物清單（JSON）       |
| `/inference`     | POST | 輸入人物 ID，取得語言模型回應       |
| `/active/update` | POST | 指定人物 ID，將其設為 active 狀態(模型Response可以接到這裡) |
| `/active/state`  | GET  | 回傳目前 active 狀態資訊（JSON） |
