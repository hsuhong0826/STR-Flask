# ===================== 導入所需套件 =====================
from flask import Flask, request, send_file, jsonify, session     # Flask 核心功能：Web框架、接收請求、回傳 JSON、session 管理
from flask_cors import CORS                                       # 處理跨來源請求（前端可跨網域存取 API）
from PIL import Image                                             # 處理圖片（開啟、轉換、儲存）
import io                                                         # 在記憶體中操作圖片（BytesIO）
import mysql.connector                                            # 連接與操作 MySQL 資料庫
from model import load_model, run_inference                       # 載入自定義模型與推論函式
from datetime import datetime, timedelta                          # 處理時間（登入錯誤紀錄等用途）
from flask import render_template

# ===================== 建立 Flask 應用 =====================
app = Flask(__name__)                     # 建立 Flask 應用實例
app.secret_key = "hong0826"              # 設定 session 加密金鑰（防止偽造登入）
CORS(app)                                # 開啟跨來源資源共享（允許前端呼叫此 API）

# ===================== MySQL 資料庫連線設定 =====================
db_config = {                             # 資料庫連線資訊
    'host': 'localhost',                 # 資料庫主機位置（本機）
    'user': 'root',                      # 資料庫使用者名稱
    'password': '',                      # 資料庫密碼（可視需求設定）
    'database': 'sketch2real'            # 使用的資料庫名稱
}

# 建立 MySQL 資料庫連線
def get_db_connection():
    return mysql.connector.connect(**db_config)

# 嘗試連線並確認資料庫是否可用
try:
    conn = get_db_connection()
    print("✅ 成功連接 MySQL")
except Exception as e:
    print("❌ 無法連接 MySQL：", e)

# ===================== 載入訓練模型（G1, G2） =====================
model = load_model(device='cpu')

# ===================== 顯示畫面 =====================
@app.route("/login", methods=["GET"])
def login_page():
    return render_template("login.html")

# 顯示 register.html 畫面
@app.route("/register", methods=["GET"])
def register_page():
    return render_template("register.html")

# 顯示 forget.html 畫面
@app.route("/forget", methods=["GET"])
def forget_page():
    return render_template("forget.html")

# 顯示 index.html 畫面（登入成功後的主畫面）
@app.route("/", methods=["GET"])
def index_page():
    return render_template("index.html")

# ===================== 推論 API =====================
@app.route("/predict", methods=["POST"])           # POST 請求處理圖片轉換
def predict():
    if 'image' not in request.files:               # 檢查是否有圖片欄位
        return "請上傳圖片", 400                    # 若無則回傳錯誤訊息

    file = request.files["image"]                  # 取得上傳的圖片檔
    try:
        image = Image.open(file.stream).convert("RGB")  # 嘗試開啟圖片並轉為 RGB 模式
    except:
        return "圖片格式錯誤", 400                   # 若開啟失敗，回傳錯誤訊息

    output = run_inference(model, image, device='cpu')  # 使用模型進行推論

    buffer = io.BytesIO()                           # 建立記憶體緩衝區
    output.save(buffer, format="PNG")               # 將圖片輸出為 PNG 格式儲存至緩衝區
    buffer.seek(0)                                  # 重設讀取位置為開頭

    return send_file(buffer, mimetype="image/png")  # 將圖片回傳給前端

# ===================== 註冊 API =====================
@app.route('/register', methods=['POST'])           # POST 請求處理註冊
def register():
    data = request.json                             # 取得前端送來的 JSON 資料
    name = data['name']                             # 使用者姓名
    username = data['username']                     # 帳號
    password = data['password']                     # 密碼

    conn = get_db_connection()                      # 建立資料庫連線
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM users WHERE username = %s", (username,))  # 檢查帳號是否已存在
    if cursor.fetchone():
        return jsonify({'msg': '帳號已存在'}), 400

    cursor.execute(                                     # 將新帳號寫入資料庫
        "INSERT INTO users (name, username, password) VALUES (%s, %s, %s)", 
        (name, username, password)
    )
    conn.commit()
    conn.close()
    return jsonify({'msg': '註冊成功'})              # 回傳成功訊息

# ===================== 登入 API =====================
@app.route('/login', methods=['POST'])               # POST 請求處理登入
def login():
    data = request.json
    username = data['username']
    password = data['password']

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    # 查詢 10 分鐘內登入失敗次數
    ten_minutes_ago = datetime.now() - timedelta(minutes=10)
    cursor.execute(
        "SELECT COUNT(*) AS fail_count FROM login_attempts WHERE username = %s AND attempt_time > %s",
        (username, ten_minutes_ago)
    )
    fail_count = cursor.fetchone()['fail_count']

    if fail_count >= 5:
        conn.close()
        return jsonify({'msg': '登入失敗次數過多', 'show_forget': True}), 400

    cursor.execute("SELECT * FROM users WHERE username = %s", (username,))  # 查詢帳號
    user = cursor.fetchone()

    if not user:
        conn.close()
        return jsonify({'msg': '帳號尚未註冊'}), 400

    if user['password'] != password:               # 密碼錯誤，記錄失敗次數
        cursor.execute("INSERT INTO login_attempts (username, attempt_time) VALUES (%s, %s)", 
                       (username, datetime.now()))
        conn.commit()
        conn.close()
        return jsonify({'msg': '密碼錯誤'}), 400

    cursor.execute("DELETE FROM login_attempts WHERE username = %s", (username,))  # 清除錯誤紀錄
    conn.commit()
    conn.close()

    session['user'] = username                     # 寫入 session
    return jsonify({"msg": "登入成功", "name": user['name']})  # 回傳登入成功與使用者姓名

# ===================== 忘記密碼 API =====================
@app.route('/forget', methods=['POST'])             # POST 請求處理找回密碼
def forget():
    data = request.json
    username = data['username']
    name = data['name']

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        "SELECT * FROM users WHERE username = %s AND name = %s", 
        (username, name)
    )
    user = cursor.fetchone()

    if user:  # 帳號與姓名符合
        cursor.execute("DELETE FROM login_attempts WHERE username = %s", (username,))  # 清除登入錯誤紀錄
        conn.commit()
        conn.close()
        return jsonify({'msg': f"您的密碼為：{user['password']}"})  # 回傳密碼
    else:
        conn.close()
        return jsonify({'msg': "帳號或姓名不正確，無法找回密碼"}), 400  # 錯誤訊息

# ===================== 啟動伺服器 =====================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)  # 啟動 Flask，監聽所有 IP 並開啟除錯模式
