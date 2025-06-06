from flask import Flask, request, jsonify, send_from_directory, render_template, Response, url_for
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
import os
import subprocess
from datetime import datetime, timezone
import cv2
import face_recognition
import numpy as np
import time
import threading

# --- GPIO (Buzzer) Setup ---
GPIO_AVAILABLE = False
BUZZER_PIN = 17  # Change to your actual BCM GPIO pin
try:
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(BUZZER_PIN, GPIO.OUT)
    GPIO.output(BUZZER_PIN, GPIO.LOW)
    GPIO_AVAILABLE = True
except Exception:
    print("WARNING: GPIO initialization failed. Buzzer is disabled.")

app = Flask(__name__)

# Configuration of photo upload folders
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# SQLite configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///./database.db'
db = SQLAlchemy(app)

#  Realtime response structure
active_state = {
    "updated_at": None,
    "person_id": None,
    "name": None,
    "description": None,
    "image_url": None,
    "response": None
}
# Global variables for face recognition
known_face_encodings = []
known_face_names = []
# Initialize video capture
video_capture = cv2.VideoCapture(0)

def load_known_face():
    folder_path = './static/uploads/'
    global known_face_encodings, known_face_names
    known_face_encodings.clear()
    known_face_names.clear()

    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder_path, filename)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)

            if len(encodings) > 0:
                known_face_encodings.append(encodings[0])
                known_face_names.append(os.path.splitext(filename)[0])

def activate_buzzer(beeps=3):
    """Activates the buzzer if GPIO is available."""
    if not GPIO_AVAILABLE: return
    print(f"Buzzer: Activating {beeps} beeps.")
    try:
        for _ in range(beeps):
            GPIO.output(BUZZER_PIN, GPIO.HIGH); time.sleep(0.15)
            GPIO.output(BUZZER_PIN, GPIO.LOW); time.sleep(0.15)
    except Exception as e:
        print(f"Buzzer Error: {e}")

def llm_forward(prompt):
    command = ["ollama", "run", "gemma:2b", prompt]
    result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=180)

    return result.stdout.strip()

# Avoid blocking the main thread with LLM calls
def update_llm_response_in_background(person_id, name, amount_owed):
    """Generates LLM response in a background thread to keep UI responsive."""
    global active_state
    prompt = (f"This person, {name}, currently owes {amount_owed}. "
              f"Generate a very short, witty, or advisory comment about this financial situation. "
              f"Keep it to one or two concise sentences.")
    llm_response = llm_forward(prompt)
    if active_state.get("person_id") == person_id:
        active_state["response"] = llm_response
        active_state["updated_at"] = datetime.now(timezone.utc).isoformat()
        print(f"Background LLM Thread: Response updated for {name}.")

def gen_frames():
    global active_state
    if not video_capture: return
    
    while True:
        success, frame = video_capture.read()
        if not success: time.sleep(0.5); continue

        # Resize for performance
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
            if True in matches:
                best_match_index = np.argmin(face_recognition.face_distance(known_face_encodings, face_encoding))
                if matches[best_match_index]:
                    recognized_name = known_face_names[best_match_index]
                    break 

        recognized_name = None
        person_to_activate = None

        if recognized_name:
            with app.app_context():
                person_in_db = Person.query.filter_by(name=recognized_name).first()
            if person_in_db and person_in_db.id != active_state.get("person_id"):
                person_to_activate = person_in_db
        
        if person_to_activate:
            print(f"Camera: New person detected: {person_to_activate.name}")
            active_state.update({
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "person_id": person_to_activate.id, "name": person_to_activate.name,
                "amount_owed": person_to_activate.amount_owed,
                "image_url": f"/image/{person_to_activate.id}",
                "response": "🤖 Generating response..."
            })
            if person_to_activate.amount_owed > 30000:
                activate_buzzer(beeps=3)
            
            threading.Thread(
                target=update_llm_response_in_background,
                args=(person_to_activate.id, person_to_activate.name, person_to_activate.amount_owed)
            ).start()

        # Draw boxes on frame for display
        for (top, right, bottom, left) in face_locations:
            top *= 4; right *= 4; bottom *= 4; left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Convert frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# data table model
class Person(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, unique=True)
    amount_owed = db.Column(db.Integer, nullable=True, default=0)
    image_path = db.Column(db.String(200))

with app.app_context():
    db.create_all()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data')
def data_page():
    return render_template('data.html')

# upload pictures & description
@app.route('/upload', methods=['POST'])
def upload():
    name = request.form.get('name')
    amount_owed_str = request.form.get('amount_owed', '0')
    image = request.files.get('image')

    if not name or not image:
        return jsonify({"status": "error", "message": "Name and image are required"}), 400
    
    try:
        amount_owed = int(amount_owed_str)
        filename = f"{secure_filename(name)}.{image.filename.rsplit('.', 1)[1].lower()}"
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(image_path)
        
        with app.app_context():
            person = Person.query.filter_by(name=name).first()
            if person:
                person.amount_owed = amount_owed
                person.image_path = filename
            else:
                person = Person(name=name, amount_owed=amount_owed, image_path=filename)
                db.session.add(person)
            db.session.commit()
        
        load_known_face()
        return jsonify({"status": "success", "person_id": person.id})
    except Exception as e:
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500

# List all people
@app.route('/list', methods=['GET'])
def list_people():
    with app.app_context(): people = Person.query.all()
    return jsonify([
        {
            "id": p.id,
            "name": p.name,
            "description": p.description,
            "image_url": f"/image/{p.id}"
        } for p in people
    ])

# show image
@app.route('/image/<int:person_id>', methods=['GET'])
def get_image(person_id):
    with app.app_context(): person = db.session.get(Person, person_id)
    if not person:
        return "Not found", 404
    return send_from_directory(app.config['UPLOAD_FOLDER'], os.path.basename(person.image_path))

@app.route('/active/update', methods=['POST'])
def update_active():
    global active_state
    data = request.json
    person_id = data.get("person_id")

    with app.app_context():
        person = db.session.get(Person, person_id)
        if not person: return jsonify({"error": "Person not found"}), 404

        print(f"Manually updating active state to: {person.name}")
        active_state.update({
            "updated_at": datetime.now(timezone.utc).isoformat(), "person_id": person.id,
            "name": person.name, "amount_owed": person.amount_owed,
            "image_url": url_for('get_image', person_id=person.id),
            "response": "🤖 Generating response..."
        })

        if person.amount_owed > 30000:
            activate_buzzer(beeps=3)
        
        threading.Thread(target=update_llm_response_in_background, args=(person.id, person.name, person.amount_owed)).start()
    return jsonify({"status": "active manually updated", "active": active_state})

@app.route('/person/update_debt/<int:person_id>', methods=['POST'])
def update_person_debt(person_id):
    """Allows updating a person's debt from the UI."""
    global active_state
    try:
        new_amount = int(request.json.get('amount_owed'))
        with app.app_context():
            person = db.session.get(Person, person_id)
            if not person: return jsonify({"status": "error", "message": "Person not found"}), 404
            
            person.amount_owed = new_amount
            db.session.commit()
            
            if active_state.get("person_id") == person_id:
                active_state["amount_owed"] = new_amount
                active_state["updated_at"] = datetime.now(timezone.utc).isoformat()
            
            return jsonify({"status": "success", "name": person.name, "new_amount_owed": new_amount})
    except Exception as e:
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/active/state', methods=['GET'])
def get_active_state():
    return jsonify(active_state)

@app.route('/active')
def show_active():
    return render_template('active.html')

@app.route('/control')
def control_panel():
    with app.app_context(): people = Person.query.all()
    return render_template("control.html", people=people)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    with app.app_context():
        db.create_all()

    load_known_face()
    try:
        app.run(host='0.0.0.0', port=8000, debug=False, threaded=True)
    finally:
        if video_capture:
            video_capture.release()
            print("Camera released.")
        if GPIO_AVAILABLE:
            GPIO.cleanup()