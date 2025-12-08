import sqlite3

conn = sqlite3.connect("users.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    phone TEXT NOT NULL UNIQUE,
    email TEXT,
    password_hash TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
)
""")

conn.commit()
conn.close()

print("Users database created successfully!")


from flask import Flask, request, jsonify
 import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
DB_PATH = "users.db"

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# -------------------------
# REGISTER USER
# -------------------------
@app.route('/register', methods=['POST'])
def register():
    data = request.json

    name = data.get("name")
    phone = data.get("phone")
    email = data.get("email")  # optional
    password = data.get("password")

    # Check mandatory fields
    if not name or not phone or not password:
        return jsonify({"status": "error", "message": "Name, phone, and password are required"}), 400

    password_hash = generate_password_hash(password)

    conn = get_db()
    cur = conn.cursor()

    try:
        cur.execute("""
            INSERT INTO users (name, phone, email, password_hash)
            VALUES (?, ?, ?, ?)
        """, (name, phone, email, password_hash))

        conn.commit()
        return jsonify({"status": "success", "message": "Registration Successful"})

    except sqlite3.IntegrityError:
        return jsonify({"status": "error", "message": "Phone or email already exists"}), 400



# -------------------------
# LOGIN USER (name + password)
# -------------------------
@app.route('/login', methods=['POST'])
def login():
    data = request.json

    name = data.get("name")
    password = data.get("password")

    if not name or not password:
        return jsonify({"status": "error", "message": "Name and password required"}), 400

    conn = get_db()
    cur = conn.cursor()

    cur.execute("SELECT * FROM users WHERE name = ?", (name,))
    user = cur.fetchone()

    if user is None:
        return jsonify({"status": "fail", "message": "Login Unsuccessful: User not found"}), 401

    if check_password_hash(user["password_hash"], password):
        return jsonify({"status": "success", "message": "Login Successful"})
    else:
        return jsonify({"status": "fail", "message": "Login Unsuccessful: Wrong password"}), 401



if __name__ == '__main__':
    app.run(debug=True)
