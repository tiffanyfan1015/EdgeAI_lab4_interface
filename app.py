from flask import Flask, request, jsonify, send_from_directory, render_template, Response
from flask_sqlalchemy import SQLAlchemy
import os
import subprocess
from datetime import datetime
import subprocess
import cv2
import face_recognition


#  Realtime response structure
active_state = {
    "updated_at": None,
    "person_id": None,
    "name": None,
    "description": None,
    "image_url": None,
    "response": None
}

app = Flask(__name__)

known_face_encodings = []
known_face_names = []

def load_known_face():
    folder_path = './static/uploads/'

    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder_path, filename)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)

            if len(encodings) > 0:
                known_face_encodings.append(encodings[0])
                known_face_names.append(os.path.splitext(filename)[0])

# Initialize video capture
video_capture = cv2.VideoCapture(0)

# Configuration of photo upload folders
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# SQLite configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///./database.db'
db = SQLAlchemy(app)

def llm_forward(img_path, command):
    ollama_command = "ollama run llava-phi3:3.8b"
    img_path = ' \"' + img_path + '\" '

    sent_command = ollama_command + img_path + command

    result = subprocess.run(sent_command, capture_output=True, text=True, shell=True)

    return result.stdout

def gen_frames():
    while True:
        success, frame = video_capture.read()
        if not success:
            break

        # Resize for performance
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = distances.argmin()

            if distances[best_match_index] < 0.5:
                name = known_face_names[best_match_index]

                # Scale back up
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw box and label
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.rectangle(frame, (left, bottom - 20), (right, bottom), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, name, (left + 4, bottom - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        # Convert frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Use MJPEG stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# data table model
class Person(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
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
    description = request.form.get('description')
    image = request.files.get('image')

    image_filename = f"{name}.jpg"
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
    image.save(image_path)

    person = Person(name=name, description=description, image_path=image_path)
    db.session.add(person)
    db.session.commit()

    return jsonify({"status": "success", "person_id": person.id})

# List all people
@app.route('/list', methods=['GET'])
def list_people():
    people = Person.query.all()
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
    person = Person.query.get(person_id)
    if not person:
        return "Not found", 404
    return send_from_directory(app.config['UPLOAD_FOLDER'], os.path.basename(person.image_path))

# inference
@app.route('/inference', methods=['POST'])
def inference():
    data = request.json
    person_id = data['person_id']
    person = Person.query.get(person_id)
    
    if not person:
        return jsonify({"error": "Person not found"}), 404

    # Retrive image path
    img_path = person.image_path
    
    # TODO: 這裡可以接 LLM ，Response或是prompt再融入一下description
    prompt = f"This is the description of a person named {person.name} :{person.description}.\n Please tell me about him/her in a humorous way."
    response = llm_forward(img_path, prompt)

    return jsonify({
        "name": person.name,
        "prompt": prompt,
        "response": response
    })

@app.route('/active/update', methods=['POST'])
def update_active():
    data = request.json
    person_id = data.get("person_id")

    if person_id == active_state.get("person_id"):
        return jsonify({"status": "no change", "active": active_state})

    person = Person.query.get(person_id)
    if not person:
        return jsonify({"error": "Person not found"}), 404

    # Prepare prompt (Todo:接上我們LLM的Response)
    img_path = person.image_path
    prompt = f"This is the description of a person named {person.name} :{person.description}.\n Please tell me about him/her in a humorous way."
    response_text = llm_forward(img_path, prompt)

    active_state.update({
        "updated_at": datetime.utcnow().isoformat(),
        "person_id": person.id,
        "name": person.name,
        "description": person.description,
        "image_url": f"/image/{person.id}",
        "response": response_text
    })

    return jsonify({"status": "active updated", "active": active_state})

@app.route('/active/state', methods=['GET'])
def get_active_state():
    return jsonify(active_state)

@app.route('/active')
def show_active():
    return render_template('active.html')

@app.route('/control')
def control_panel():
    people = Person.query.all()
    return render_template("control.html", people=people)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    load_known_face()
    app.run(debug=True, host='0.0.0.0', port=8000)
