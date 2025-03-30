import mysql.connector
from mysql.connector import Error
import numpy as np
import face_recognition
from werkzeug.security import generate_password_hash, check_password_hash
import os
from dotenv import load_dotenv

load_dotenv()

def get_db_connection():
    return mysql.connector.connect(
        host=os.getenv('localhost'),
        user=os.getenv('root'),
        password=os.getenv(''),
        database=os.getenv('db_ia')
    )

def init_db():
    try:
        # Créer la base de données si elle n'existe pas
        conn = mysql.connector.connect(
            host=os.getenv('MYSQL_HOST'),
            user=os.getenv('MYSQL_USER'),
            password=os.getenv('MYSQL_PASSWORD')
        )
        cursor = conn.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {os.getenv('MYSQL_DATABASE')}")
        cursor.close()
        conn.close()

        # Se connecter à la base de données et créer la table
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(255) UNIQUE NOT NULL,
                email VARCHAR(255) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                face_encoding LONGBLOB,
                google_id VARCHAR(255) UNIQUE,
                facebook_id VARCHAR(255) UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
    except Error as e:
        print(f"Error: {e}")
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

def add_user(username, email, password, face_encoding=None, google_id=None, facebook_id=None):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        password_hash = generate_password_hash(password)
        face_encoding_bytes = face_encoding.tobytes() if face_encoding is not None else None
        
        sql = '''
            INSERT INTO users (username, email, password_hash, face_encoding, google_id, facebook_id)
            VALUES (%s, %s, %s, %s, %s, %s)
        '''
        values = (username, email, password_hash, face_encoding_bytes, google_id, facebook_id)
        
        cursor.execute(sql, values)
        conn.commit()
        return True
    except Error as e:
        print(f"Error: {e}")
        return False
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

def verify_user(username, password):
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute('SELECT * FROM users WHERE username = %s', (username,))
        user = cursor.fetchone()
        
        if user and check_password_hash(user['password_hash'], password):
            return {
                'id': user['id'],
                'username': user['username'],
                'email': user['email']
            }
        
        return None
    except Error as e:
        print(f"Error: {e}")
        return None
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

def verify_face_login(face_encoding):
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute('SELECT id, username, email, face_encoding FROM users WHERE face_encoding IS NOT NULL')
        users = cursor.fetchall()
        
        for user in users:
            stored_encoding = np.frombuffer(user['face_encoding'])
            if face_recognition.compare_faces([stored_encoding], face_encoding)[0]:
                return {
                    'id': user['id'],
                    'username': user['username'],
                    'email': user['email']
                }
        
        return None
    except Error as e:
        print(f"Error: {e}")
        return None
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

def get_or_create_social_user(social_id, email, username, provider='google'):
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Vérifier si l'utilisateur existe déjà
        if provider == 'google':
            cursor.execute('SELECT * FROM users WHERE google_id = %s', (social_id,))
        else:
            cursor.execute('SELECT * FROM users WHERE facebook_id = %s', (social_id,))
        
        user = cursor.fetchone()
        
        if user:
            return {
                'id': user['id'],
                'username': user['username'],
                'email': user['email']
            }
        
        # Créer un nouveau compte
        password_hash = generate_password_hash(social_id)
        
        if provider == 'google':
            sql = '''
                INSERT INTO users (username, email, password_hash, google_id)
                VALUES (%s, %s, %s, %s)
            '''
        else:
            sql = '''
                INSERT INTO users (username, email, password_hash, facebook_id)
                VALUES (%s, %s, %s, %s)
            '''
        
        values = (username, email, password_hash, social_id)
        cursor.execute(sql, values)
        conn.commit()
        
        return {
            'id': cursor.lastrowid,
            'username': username,
            'email': email
        }
    except Error as e:
        print(f"Error: {e}")
        return None
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()