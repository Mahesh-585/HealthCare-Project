# ============================================
#   PraanAI — Flask Backend API
#   Team PraanAI | AI for Healthcare
# ============================================

from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO, emit
import pickle
import numpy as np
import sqlite3
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'praanai_secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# ── Load ML Model ─────────────────────────────
print("🔵 Loading PraanAI Model...")
model = pickle.load(open('model/triage_model.pkl', 'rb'))
le    = pickle.load(open('model/label_encoder.pkl', 'rb'))
print("✅ Model Loaded!")

# ── ESI Label Mapping ─────────────────────────
ESI_LABELS = {
    0: {'level': 2, 'label': 'EMERGENT',    'color': 'orange'},
    1: {'level': 3, 'label': 'URGENT',      'color': 'yellow'},
    2: {'level': 4, 'label': 'LESS URGENT', 'color': 'green'},
    3: {'level': 5, 'label': 'NON URGENT',  'color': 'gray'},
}

# ── Fallback Values (if device not available) ─
FALLBACKS = {
    'sbp':        120,
    'dbp':        80,
    'hr':         80,
    'rr':         18,
    'bt':         36.6,
    'saturation': 97,
}

# ── High Risk Fields ───────────────────────────
HIGH_RISK_FIELDS = ['sbp', 'dbp', 'saturation']

# ── Setup Database ────────────────────────────
def init_db():
    conn = sqlite3.connect('praanai.db')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            name           TEXT,
            age            INTEGER,
            chief_complain TEXT,
            pain           INTEGER,
            sbp            REAL,
            dbp            REAL,
            hr             REAL,
            saturation     REAL,
            esi_level      INTEGER,
            esi_label      TEXT,
            na_fields      TEXT,
            warning_level  TEXT,
            timestamp      DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()
    print("✅ Database Ready!")

init_db()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/search')
def search():
    return render_template('search.html')

@app.route('/reports')
def reports():
    return render_template('reports.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        na_fields     = data.get('na_fields', [])
        used_fallback = []

        def get_val(key):
            val = data.get(key)
            if val is None or str(val).strip() == '':
                used_fallback.append(key)
                return FALLBACKS[key]
            try:
                return float(val)
            except:
                used_fallback.append(key)
                return FALLBACKS[key]

        complaint = data.get('chief_complain', 'pain')
        complaint_encoded = le.transform([complaint])[0] if complaint in le.classes_ else 0

        # Base vitals
        age        = float(data.get('age', 30))
        arrival    = float(data.get('arrival_mode', 1))
        pain       = float(data.get('pain', 1))
        nrs_pain   = float(data.get('nrs_pain', 5))
        sbp        = get_val('sbp')
        dbp        = get_val('dbp')
        hr         = get_val('hr')
        rr         = get_val('rr')
        bt         = get_val('bt')
        saturation = get_val('saturation')

        # Engineered features (same as train_model.py)
        pulse_pressure = sbp - dbp
        shock_index    = hr / (sbp if sbp > 0 else 1)
        fever          = 1 if bt > 38 else 0
        low_spo2       = 1 if saturation < 94 else 0
        high_pain      = 1 if nrs_pain > 7 else 0
        hypertension   = 1 if sbp > 140 else 0
        hypotension    = 1 if sbp < 90 else 0
        tachycardia    = 1 if hr > 100 else 0

        features = np.array([[
            age, arrival, float(complaint_encoded), pain, nrs_pain,
            sbp, dbp, hr, rr, bt, saturation,
            pulse_pressure, shock_index, fever, low_spo2,
            high_pain, hypertension, hypotension, tachycardia
        ]])

        prediction    = model.predict(features)[0]
        esi_info      = ESI_LABELS[int(prediction)]
        all_fallbacks = list(set(na_fields + used_fallback))
        fallback_note = ', '.join(all_fallbacks).upper() if all_fallbacks else 'None'

        # ── Smart Warning System ──────────────────
        high_risk_used = [f for f in all_fallbacks if f in HIGH_RISK_FIELDS]
        low_risk_used  = [f for f in all_fallbacks if f not in HIGH_RISK_FIELDS]

        if high_risk_used:
            warning = (
                f"VERIFY MANUALLY: "
                f"{', '.join(high_risk_used).upper()} was not measured. "
                f"Fallback (average) value was used. "
                f"AI decision may NOT be accurate — doctor must confirm before treatment."
            )
            warning_level = 'high'
        elif low_risk_used:
            warning = (
                f"Note: {', '.join(low_risk_used).upper()} used fallback average value. "
                f"These are low-risk fields — AI decision is still reliable."
            )
            warning_level = 'low'
        else:
            warning       = None
            warning_level = None

        print(f"Patient: {data.get('name')} | ESI: {esi_info['level']} | Fallbacks: {fallback_note} | Warning: {warning_level or 'None'}")

        conn = sqlite3.connect('praanai.db')
        conn.execute('''
            INSERT INTO patients
            (name, age, chief_complain, pain, sbp, dbp, hr,
             saturation, esi_level, esi_label, na_fields, warning_level)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)''',
            (
                data.get('name'), data.get('age'), data.get('chief_complain'),
                data.get('pain'), get_val('sbp'), get_val('dbp'), get_val('hr'),
                get_val('saturation'), esi_info['level'], esi_info['label'],
                fallback_note, warning_level or 'none',
            ))
        conn.commit()
        conn.close()

        result = {
            'name':          data.get('name'),
            'age':           data.get('age'),
            'complaint':     data.get('chief_complain'),
            'esi_level':     esi_info['level'],
            'esi_label':     esi_info['label'],
            'color':         esi_info['color'],
            'fallbacks':     all_fallbacks,
            'warning':       warning,
            'warning_level': warning_level,
        }

        socketio.emit('new_patient', result)
        return jsonify({'success': True, 'result': result})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/patients', methods=['GET'])
def get_patients():
    conn = sqlite3.connect('praanai.db')
    cursor = conn.execute('''
        SELECT name, age, chief_complain, esi_level, esi_label,
               timestamp, na_fields, warning_level
        FROM patients ORDER BY esi_level ASC, timestamp ASC
    ''')
    patients = []
    for row in cursor.fetchall():
        patients.append({
            'name':          row[0],
            'age':           row[1],
            'complaint':     row[2],
            'esi_level':     row[3],
            'esi_label':     row[4],
            'timestamp':     row[5],
            'na_fields':     row[6],
            'warning_level': row[7],
        })
    conn.close()
    return jsonify(patients)

if __name__ == '__main__':
    print("PraanAI Server Starting...")
    print("Patient Form  → http://localhost:5000")
    print("Dashboard     → http://localhost:5000/dashboard")
    socketio.run(app, debug=True, port=5000)