import os
import psycopg2
from imgbeddings import imgbeddings
from PIL import Image

# Database connection details
DB_CONNECTION_STRING = "postgresql://pushkar1713:wNWD9PlSQkG5@ep-steep-voice-13899759.us-east-2.aws.neon.tech/python?sslmode=require"
STORED_FACES_DIR = "/Users/pushkar1713/Projects/mait-hackathon/stored-faces"

def init_db():
    conn = psycopg2.connect(DB_CONNECTION_STRING)
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
    
    conn.commit()
    cur.close()
    conn.close()

def populate_database():
    # Initialize database first
    init_db()
    
    # Connect to the database
    conn = psycopg2.connect(DB_CONNECTION_STRING)
    cur = conn.cursor()
    
    # Load the imgbeddings model
    ibed = imgbeddings()
    
    # Iterate through the images
    for filename in os.listdir(STORED_FACES_DIR):
        filepath = os.path.join(STORED_FACES_DIR, filename)
        
        if os.path.isfile(filepath):
            img = Image.open(filepath)
            embedding = ibed.to_embeddings(img)
            
            # Convert embedding to vector format
            embedding_list = embedding[0].tolist()
            
            # Insert with explicit vector cast
            cur.execute("""
                INSERT INTO pictures (picture, embedding) 
                VALUES (%s, %s::vector)
            """, (filename, embedding_list))
            
            print(f"Inserted: {filename}")
    
    conn.commit()
    cur.close()
    conn.close()
    print("Database populated successfully.")

if __name__ == "__main__":
    populate_database()
