from flask import Flask, render_template, Response
import cv2
import face_recognition

app = Flask(__name__)

# Load known face(s)
tony_image = face_recognition.load_image_file("tony.jpg")
tony_encoding = face_recognition.face_encodings(tony_image)[0]
known_face_encodings = [tony_encoding]
known_face_names = ["Tony"]

# Initialize video capture
video_capture = cv2.VideoCapture(0)

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


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)