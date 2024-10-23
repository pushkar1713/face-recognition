import cv2
import os
import numpy as np
from imgbeddings import imgbeddings
from PIL import Image
import psycopg2
from contextlib import closing  # For context manager

# Specify the path to the algorithm
alg = "haarcascade_frontalface_default.xml"
def face_to_embed():
    # Connecting to the database
    conn = psycopg2.connect("postgresql://pushkar1713:wNWD9PlSQkG5@ep-steep-voice-13899759.us-east-2.aws.neon.tech/python?sslmode=require")
    cur = conn.cursor()

        # Create vector extension
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    
    # Drop existing table
    cur.execute("DROP TABLE IF EXISTS pictures;")
    
    # Create table with correct vector dimensions (768)
    cur.execute("""
        CREATE TABLE pictures (
            picture TEXT PRIMARY KEY,
            embedding vector(768)
        );
    """)
    
    for filename in os.listdir("stored-faces"):
        filepath = os.path.join("stored-faces", filename)
        # Check if the item is a file, not a directory
        if os.path.isfile(filepath):
            # Opening the image
            img = Image.open(filepath)
            # Loading the `imgbeddings`
            ibed = imgbeddings()
            # Calculating the embeddings
            embedding = ibed.to_embeddings(img)
            cur.execute("INSERT INTO pictures VALUES (%s, %s)", (filename, embedding[0].tolist()))
            print(filename)

    # Commit the changes and close the connection
    conn.commit()
    cur.close()

def face_detection():
    # Passing the algorithm to OpenCV
    haar_cascade = cv2.CascadeClassifier(alg)

    # Specify the file path of the image
    file_name = "/Users/pushkar1713/Projects/mait-hackathon/group.jpg"
    print("Absolute path to image:", os.path.abspath(file_name))
    
    # Load the image
    img = cv2.imread(file_name)

    # Check if the image is loaded correctly
    if img is None:
        print("Error: Image not loaded correctly.")
        return

    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = haar_cascade.detectMultiScale(
        gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
    )

    # Check if any faces are detected
    if len(faces) == 0:
        print("No faces detected.")
        return

    # Draw rectangles around detected faces and save cropped images
    if not os.path.exists("stored-faces"):
        os.makedirs("stored-faces")

    for i, (x, y, w, h) in enumerate(faces):
        cropped_image = img[y:y + h, x:x + w]
        target_file_name = f'stored-faces/{i}.jpg'
        cv2.imwrite(target_file_name, cropped_image)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show the image with rectangles around detected faces
    cv2.imshow("Detected Faces", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def calc_embeds():
    # loading the face image path into file_name variable
    file_name = "/Users/pushkar1713/Projects/mait-hackathon/shahrukh_khan.jpg"  # replace with the path to your image
    # opening the image
    img = Image.open(file_name)
    # loading the `imgbeddings`
    ibed = imgbeddings()
    # calculating the embeddings
    embedding = ibed.to_embeddings(img)
    return embedding  # Return the embedding for use in `find_face`


def find_face(embedding):
    conn = psycopg2.connect("postgresql://pushkar1713:wNWD9PlSQkG5@ep-steep-voice-13899759.us-east-2.aws.neon.tech/python?sslmode=require")
    cur = conn.cursor()
    
    # Convert the NumPy array to a list of native Python floats
    
    string_representation = "["+ ",".join(str(x) for x in embedding[0].tolist()) +"]"
    cur.execute("SELECT * FROM pictures ORDER BY embedding <-> %s LIMIT 1;", (string_representation,))
    rows = cur.fetchall()
    for row in rows:
        image_path = "stored-faces/" + row[0]
        # Use OpenCV to load and display the image
        img = cv2.imread(image_path)
        if img is not None:
            cv2.imshow("Found Face", img)
            cv2.waitKey(0)  # Wait until a key is pressed
            cv2.destroyAllWindows()  # Close the image window
        else:
            print(f"Error loading image: {image_path}")

    cur.close()
    
# Call the functions
face_detection()
face_to_embed()
embedding = calc_embeds()  # Capture the embedding to find a face later
find_face(embedding)  # Pass the embedding to find_face function
