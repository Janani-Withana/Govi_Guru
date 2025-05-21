from flask import Flask
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, firestore

def create_app():
    app = Flask(__name__)
    CORS(app)

    # Firebase setup
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)
    app.db = firestore.client()

    # Register routes
    from app.routes.chat_routes import chat_bp
    app.register_blueprint(chat_bp, url_prefix="/api")

    return app

