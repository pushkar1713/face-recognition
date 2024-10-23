from flask import Flask, request, jsonify
import cv2
import os
import numpy as np
from imgbeddings import imgbeddings
from PIL import Image
import psycopg2
import io

app = Flask(__name__)

# Specify the path to the algorithm
alg = "haarcascade_frontalface_default.xml"
haar_cascade = cv2.CascadeClassifier(alg)

# Database configuration
DB_CONNECTION = "postgresql://pushkar1713:wNWD9PlSQkG5@ep-steep-voice-13899759.us-east-2.aws.neon.tech/python?sslmode=require"

def init_db():
    """Initialize database with vector extension and required table"""
    conn = psycopg2.connect(DB_CONNECTION)
    cur = conn.cursor()
    
    # Create vector extension
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    
    # Create table with vector type
    cur.execute("""
        CREATE TABLE IF NOT EXISTS pictures (
            id SERIAL PRIMARY KEY,
            picture TEXT NOT NULL,
            embedding vector(768)
        );
    """)
    
    conn.commit()
    cur.close()
    conn.close()

def get_db_connection():
    return psycopg2.connect(DB_CONNECTION)

# Initialize database when starting the application
init_db()

# Similarity threshold for face matching
SIMILARITY_THRESHOLD = 0.75

@app.route('/')
def home():
    return "Face Detection API is running!"

@app.route('/detect', methods=['POST'])
def detect_faces():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes))

    # Convert image to OpenCV format and grayscale
    open_cv_image = np.array(img.convert('RGB')) 
    gray_img = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = haar_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    if len(faces) == 0:
        return jsonify({"error": "No faces detected"}), 400

    # Crop faces, save them, and generate embeddings
    face_embeddings = []
    
    for i, (x, y, w, h) in enumerate(faces):
        cropped_image = open_cv_image[y:y + h, x:x + w]
        face_img = Image.fromarray(cropped_image)

        # Get embeddings
        ibed = imgbeddings()
        embedding = ibed.to_embeddings(face_img)
        embedding_list = embedding[0].tolist()

        # Store embedding in DB with vector type
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute("""
            INSERT INTO pictures (picture, embedding) 
            VALUES (%s, %s::vector)
        """, (f'face_{i}.jpg', embedding_list))
        
        face_embeddings.append(embedding_list)
        
        conn.commit()
        cur.close()
        conn.close()

    return jsonify({
        "faces_detected": len(faces),
        "embeddings": face_embeddings
    })

@app.route('/match', methods=['POST'])
def match_face():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes))

    # Calculate embedding
    ibed = imgbeddings()
    embedding = ibed.to_embeddings(img)
    embedding_list = embedding[0].tolist()

    # Find the closest match in the database
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Query using vector comparison with similarity threshold
    cur.execute("""
        SELECT picture, 1 - (embedding <=> %s::vector) as similarity
        FROM pictures 
        WHERE 1 - (embedding <=> %s::vector) >= %s
        ORDER BY similarity DESC 
        LIMIT 1;
    """, (embedding_list, embedding_list, SIMILARITY_THRESHOLD))
    
    result = cur.fetchone()
    cur.close()
    conn.close()

    if result:
        return jsonify({
            "matched_face": result[0],
            "similarity": float(result[1])
        })
    else:
        return jsonify({"error": "No match found"}), 404

if __name__ == '__main__':
    app.run(debug=True)