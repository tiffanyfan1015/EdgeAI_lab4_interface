from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_sqlalchemy import SQLAlchemy
import os
from datetime import datetime

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

# Configuration of photo upload folders
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# SQLite configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///./database.db'
db = SQLAlchemy(app)

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

    # TODO: 這裡可以接 LLM 推論
    prompt = f"這是關於 {person.name} 的描述：\n{person.description}\n請幫我想一句毒舌歡迎語。"
    response = f"喔，又是你啊，{person.name}，你的臉我永遠忘不了（因為太難忘了）。"

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
    prompt = f"這是關於 {person.name} 的描述：{person.description}\n請幫我想一句毒舌歡迎語。"
    response_text = f"{person.name}，你終於來了，我還以為你在社交黑洞裡迷路了。"

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


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
