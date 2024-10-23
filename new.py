from flask import Flask, request, jsonify
import cv2
from PIL import Image
import numpy as np
from imgbeddings import imgbeddings
import psycopg2

app = Flask(__name__)

def get_db_connection():
    return psycopg2.connect("postgresql://pushkar1713:wNWD9PlSQkG5@ep-steep-voice-13899759.us-east-2.aws.neon.tech/python?sslmode=require")

def compare_faces(face_image):
    # Convert NumPy array to PIL image
    pil_face_image = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
    ibed = imgbeddings()
    embedding = ibed.to_embeddings(pil_face_image)
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Convert embedding to list
    embedding_list = embedding[0].tolist()
    
    # Query using vector comparison with cosine distance
    cur.execute("""
        SELECT picture, embedding <=> %s::vector AS distance 
        FROM pictures 
        ORDER BY distance ASC 
        LIMIT 1;
    """, (embedding_list,))
    
    row = cur.fetchone()
    cur.close()
    conn.close()
    
    if row:
        return row[0]
    return None

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files['image']
    img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Error processing the image"}), 400

    haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = haar_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return jsonify({"error": "No faces detected."}), 400

    matches = []
    for (x, y, w, h) in faces:
        face_image = img[y:y + h, x:x + w]
        matched_name = compare_faces(face_image)
        if matched_name:
            matches.append(matched_name)

    return jsonify({"matches": matches})

if __name__ == '__main__':
    app.run(debug=True)