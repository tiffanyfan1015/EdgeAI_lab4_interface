# final_app_with_inference.py
# This script re-adds the /inference route to allow on-demand LLM generation from the data page.

# --- Imports ---
from flask import Flask, render_template, Response, request, jsonify, url_for, send_from_directory
from flask_sqlalchemy import SQLAlchemy
import cv2
import time
import os
import face_recognition
import numpy as np
from werkzeug.utils import secure_filename
from datetime import datetime, timezone
import threading
import traceback
import requests
import json


# --- Flask App Initialization and Configuration ---
app = Flask(__name__)
app.secret_key = "a_very_final_secret_key_v3"

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///./database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- Global Variables ---
known_face_encodings = []
known_face_names = []
active_state = {
    "updated_at": None, "person_id": None, "name": "N/A", 
    "amount_owed": 0, "image_url": None, "response": "Waiting...", 
    "description": ""
}

# --- Camera Initialization ---
camera = None
print("Program starting: Attempting to initialize camera...")
try:
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print(f"!!! ERROR: Could not open camera. !!!"); camera = None
    else:
        print(f"Camera successfully opened.")
except Exception as e:
    print(f"!!! Camera init exception: {e} !!!"); camera = None

# --- Database Model Definition ---
class Person(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, unique=True)
    amount_owed = db.Column(db.Integer, nullable=True, default=0)
    image_path = db.Column(db.String(200), nullable=True)
    description = db.Column(db.Text, nullable=True)
    llm_response = db.Column(db.Text, nullable=True)

# --- Helper Functions ---
def load_known_face():
    folder_path = './static/uploads/'

    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            name = (os.path.splitext(filename)[0]).capitalize()
            existing_person = Person.query.filter_by(name=name).first()

            if existing_person:
                continue

            image_path = os.path.join(folder_path, filename)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)

            if len(encodings) > 0:
                known_face_encodings.append(encodings[0])
                known_face_names.append(name)

                person = Person(name=name, description="my classmate and he is handsome, currently owes me 100 dollars.", image_path=image_path, amount_owed=100)
                db.session.add(person)
                db.session.commit()


def allowed_file(filename): 
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def llm_forward(prompt):
    """
    Calls the local Ollama REST API to generate a response from a text-based model.
    This is more efficient than using subprocess for every call.
    """
    # The default endpoint for the local Ollama server
    ollama_api_endpoint = "http://127.0.0.1:11434/api/generate"
    
    # The name of the text-only model you want to use (e.g., gemma:2b)
    model_name = "tinyllama" 

    # The payload (data) to send to the API, in JSON format
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False  # Set to False to get the full response at once, which is simpler
    }

    try:
        # Send the POST request to the Ollama API
        response = requests.post(
            ollama_api_endpoint,
            json=payload,
            timeout=180  # 3-minute timeout for the API call
        )
        
        # Check for HTTP errors (e.g., 404 Not Found, 500 Internal Server Error)
        response.raise_for_status()
        
        # The response from Ollama (with stream:false) is a single JSON object.
        # We need to parse it to get the 'response' field.
        response_data = response.json()
        return response_data.get("response", "").strip()

    except requests.exceptions.Timeout:
        print("  Error: Request to local Ollama API timed out (3 minutes).")
        return "LLM request timed out."
    
    except requests.exceptions.ConnectionError:
        print(f"  Error: Could not connect to Ollama server at {ollama_api_endpoint}")
        return "LLM server is not reachable. Is Ollama running?"
    
    except requests.exceptions.RequestException as e:
        print(f"  Error calling local Ollama API: {e}")
        return f"LLM API Error: {e}"
    
    except json.JSONDecodeError:
        print("  Error: Could not decode JSON response from Ollama.")
        return "LLM returned an invalid response."


def generate_and_save_llm_response(person_id):
    """
    Fetches a person, generates an LLM response based on their data,
    and saves it back to the database. Runs in a background thread.
    """
    with app.app_context():
        person = db.session.get(Person, person_id)
        if not person:
            print(f"Background Thread: Person with ID {person_id} not found.")
            return

        print(f"Background Thread: Generating LLM response for {person.name}...")
        prompt = (f"This person, {person.name}, who is {person.description}, currently owes me {person.amount_owed}. "
                  f"Generate a very short, witty sentence to remind me asking for my money in about 50 words.")

        llm_response_text = llm_forward(prompt)

        person.llm_response = llm_response_text
        db.session.commit()
        print(f"Background Thread: Saved new LLM response for {person.name} to DB.")

# --- Video Stream Generation Logic ---
def generate_frames():
    # ... (This function remains the same as the previous version) ...

    if camera is None or not camera.isOpened():
        print("generate_frames: Camera not available.")
        return
    
    while True:
        try:
            success, frame = camera.read()
            if not success or frame is None: 
                time.sleep(0.5)
                continue
            
            frame = cv2.flip(frame, 1)

            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names_in_frame = []
            recognized_name = None
            
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
                name = "Unknown"

                if True in matches and known_face_encodings:
                    best_match_index = np.argmin(face_recognition.face_distance(known_face_encodings, face_encoding))
                    
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]
                        
                        if recognized_name is None: 
                            recognized_name = name

                face_names_in_frame.append(name)
            
            person_object_for_trigger = None

            if recognized_name:
                with app.app_context():
                    person_in_db = Person.query.filter_by(name=recognized_name).first()

                if person_in_db and person_in_db.id != active_state.get("person_id"):
                    person_object_for_trigger = person_in_db

            for (top, right, bottom, left), name_to_draw in zip(face_locations, face_names_in_frame):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
                
                font = cv2.FONT_HERSHEY_DUPLEX; cv2.putText(frame, name_to_draw, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)
            
            if person_object_for_trigger:
                active_state.update({
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "person_id": person_object_for_trigger.id,
                    "name": person_object_for_trigger.name,
                    "amount_owed": person_object_for_trigger.amount_owed,
                    "description": person_object_for_trigger.description, # Add this line
                    "image_url": f"/image/{person_object_for_trigger.id}",
                    "response": "🤖 Generating response..."
                })
            
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret: 
                continue

            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        except Exception as e: 
            print(f"generate_frames: Error in loop: {e}")
            traceback.print_exc()
            break

    print("generate_frames: Video stream stopped.")

# --- Flask Routes ---
@app.route('/')
def index(): 
    return render_template('index.html')

@app.route('/active')
def active_html_page(): 
    return render_template('active.html')

@app.route('/webcam')
def webcam_html_page(): 
    return render_template('webcam.html')

@app.route('/control')
def control_html_page():
    people = []

    try:
        with app.app_context(): 
            people = Person.query.all()
    except Exception as e: 
        print(f"Error reading people for /control: {e}")

    return render_template('control.html', people=people) 

@app.route('/video_feed')
def video_feed():
    if camera is None or not camera.isOpened(): 
        return "Camera not available", 503
    
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/data')
def data_page_route(): 
    return render_template('data.html')

@app.route('/person/update_debt/<int:person_id>', methods=['POST'])
def update_person_debt_route(person_id):
    global active_state; data = request.json; new_amount_str = data.get('amount_owed')

    if new_amount_str is None: 
        return jsonify({"status": "error", "message": "amount_owed is required"}), 400
    
    try: 
        new_amount = int(new_amount_str)
    except ValueError: 
        return jsonify({"status": "error", "message": "amount_owed must be a number"}), 400
    
    try:
        with app.app_context():
            person = db.session.get(Person, person_id)
            if not person: 
                return jsonify({"status": "error", "message": "Person not found"}), 404
            
            person.amount_owed = new_amount; db.session.commit()
            print(f"Updated {person.name}'s amount_owed to {new_amount} in DB.")
            
            if active_state.get("person_id") == person_id:
                active_state["amount_owed"] = new_amount; active_state["updated_at"] = datetime.now(timezone.utc).isoformat()
                print(f"Updated active_state for {person.name} with new amount.")
            threading.Thread(target=generate_and_save_llm_response, args=(person.id,)).start()
            return jsonify({"status": "success", "name": person.name, "new_amount_owed": new_amount})
    
    except Exception as e: 
        db.session.rollback()
        return jsonify({"status": "error", "message": f"Database error: {e}"}), 500

@app.route('/person/update_description/<int:person_id>', methods=['POST'])
def update_person_description_route(person_id):
    data = request.json
    new_description = data.get('description')

    if new_description is None:
        return jsonify({"status": "error", "message": "description is required"}), 400

    try:
        with app.app_context():
            person = db.session.get(Person, person_id)
            if not person:
                return jsonify({"status": "error", "message": "Person not found"}), 404

            person.description = new_description
            db.session.commit()
            print(f"Updated {person.name}'s description in DB.")

            # Update description in active_state as well if they are the active person
            if active_state.get("person_id") == person_id:
                active_state["description"] = new_description
                active_state["updated_at"] = datetime.now(timezone.utc).isoformat()
                print(f"Updated active_state for {person.name} with new description.")
            threading.Thread(target=generate_and_save_llm_response, args=(person.id,)).start()
            return jsonify({"status": "success", "name": person.name, "new_description": new_description})

    except Exception as e:
        db.session.rollback()
        return jsonify({"status": "error", "message": f"Database error: {e}"}), 500

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files: 
        return jsonify({"status": "error", "message": "No image file"}), 400
    
    name = request.form.get('name')
    amount_owed_str = request.form.get('amount_owed')
    description = request.form.get('description', '') # Use a default empty string
    image_file = request.files['image']
    
    if not name or image_file.filename == '': 
        return jsonify({"status": "error", "message": "Name and image are required"}), 400
    
    try:
        with app.app_context():
            # Check if person already exists
            existing_person = Person.query.filter_by(name=name).first()
            if existing_person: return jsonify({"status": "error", "message": f"Person with name '{name}' already exists."}), 409
            
            image_file = request.files['image']
            filename = f"{secure_filename(name)}.{image_file.filename.rsplit('.', 1)[1].lower()}"
            image_path_full = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image_file.save(image_path_full)

            new_person = Person(
                name=name,
                amount_owed=int(request.form.get('amount_owed', 0)),
                description=request.form.get('description', ''),
                image_path=filename,
                llm_response="Generating first response..." # Initial placeholder
            )
            db.session.add(new_person); db.session.commit()
            
            # Add new person to in-memory face recognition
            image = face_recognition.load_image_file(image_path_full)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(new_person.name)
            
            # Trigger generation in the background
            threading.Thread(target=generate_and_save_llm_response, args=(new_person.id,)).start()
            return jsonify({"status": "success", "person_id": new_person.id})

    except Exception as e:
        db.session.rollback(); return jsonify({"status": "error", "message": f"Upload failed: {e}"}), 500

# List all people
@app.route('/list', methods=['GET'])
def list_people():
    people = Person.query.all()
    return jsonify([
        {
            "id": p.id,
            "name": p.name,
            "description": p.description,
            "amount_owed": p.amount_owed,
            "image_url": f"/image/{p.id}"
        } for p in people
    ])

# show image
@app.route('/image/<int:person_id>', methods=['GET'])
def get_image(person_id):
    person = Person.query.get(person_id)
    if not person:
        return "Not found", 404

    return send_from_directory(app.config['UPLOAD_FOLDER'], os.path.basename(person.image_path))

@app.route('/active/state', methods=['GET'])
def get_active_state_route(): 
    return jsonify(active_state)

@app.route('/active/update', methods=['POST'])
def update_active_route():
    global active_state 
    data = request.json
    person_id = data.get("person_id")
    
    if person_id is None: 
        return jsonify({"status": "error", "message": "person_id is required"}), 400
    
    with app.app_context():
        person = db.session.get(Person, person_id)
        
        if not person: 
            return jsonify({"status": "error", "message": "Person not found"}), 404
    
        print(f"  Manually updating active_state to: {person.name}, Owes: {person.amount_owed}.")

        active_state.update({
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "person_id": person.id,
            "name": person.name,
            "amount_owed": person.amount_owed,
            "description": person.description, # Add this line
            "image_url": url_for('get_image', person_id=person.id),
            "response": "🤖 Generating response..."
        })
    
    return jsonify({"status": "active manually updated", "active": active_state})

# --- NEW: Re-added /inference route for the button on data.html ---
@app.route('/inference', methods=['POST'])
def inference_route():
    """Handles on-demand LLM inference requests from the data.html page."""

    data = request.json
    person_id = data.get('person_id')
    print(f"Route: /inference - Received LLM request for person_id: {person_id}")

    if person_id is None:
        return jsonify({"error": "person_id is required", "response": "Error: Missing person_id."}), 400

    generate_and_save_llm_response(person_id)

    with app.app_context():
        person = db.session.get(Person, person_id)
        if not person: return jsonify({"error": "Person not found"}), 404
        return jsonify({"name": person.name, "response": person.llm_response})
    
    # if not person:
    #     return jsonify({"error": "Person not found", "response": f"Error: Person with ID {person_id} not found."}), 404

    # # The prompt is based on name and debt amount, using the text-only LLM
    # prompt = (f"This person, {person.name}, who is {person.description}, currently owes me {person.amount_owed}. "
    #             f"Generate a very short, witty sentence to remind me asking for my money in about 50 words.")
    
    # llm_response_text = llm_forward(prompt)
    
    # return jsonify({
    #     "name": person.name,
    #     "response": llm_response_text 
    # })

# --- Main Application Execution ---
if __name__ == '__main__':
    try:
        with app.app_context():
            db.create_all()
            db.session.commit()
            load_known_face() 
            print("Database tables checked/created.")
        
        print("Reminder: Ensure Ollama service is running and model (e.g., 'gemma:2b') is downloaded.")
        
        print("Starting Flask application...")
        app.run(host='0.0.0.0', port=8000, debug=False, threaded=True) 

    except KeyboardInterrupt: 
        print("\nApplication interrupted by user (Ctrl+C).")
    except Exception as e: 
        print(f"An error occurred: {e}")
    
    finally:
        if camera is not None and camera.isOpened(): 
            print("Releasing camera...")
            camera.release()
    
        print("Application shut down.")
