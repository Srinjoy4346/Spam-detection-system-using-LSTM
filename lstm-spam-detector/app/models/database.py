# app/models/database.py
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()

class User(UserMixin, db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    
    # Relationships
    emails = db.relationship('Email', backref='user', lazy=True)
    predictions = db.relationship('Prediction', backref='user', lazy=True)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def __repr__(self):
        return f'<User {self.username}>'

class Email(db.Model):
    __tablename__ = 'emails'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    subject = db.Column(db.Text)
    content = db.Column(db.Text, nullable=False)
    sender = db.Column(db.String(255))
    recipient = db.Column(db.String(255))
    received_date = db.Column(db.DateTime, default=datetime.utcnow)
    actual_label = db.Column(db.String(10))  # 'spam' or 'ham', if known
    
    # Relationships
    predictions = db.relationship('Prediction', backref='email', lazy=True)
    
    def __repr__(self):
        return f'<Email {self.id}: {self.subject[:50]}>'

class Prediction(db.Model):
    __tablename__ = 'predictions'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    email_id = db.Column(db.Integer, db.ForeignKey('emails.id'), nullable=False)
    predicted_label = db.Column(db.String(10), nullable=False)  # 'spam' or 'ham'
    confidence_score = db.Column(db.Float, nullable=False)
    model_version = db.Column(db.String(50), default='v1.0')
    prediction_time = db.Column(db.DateTime, default=datetime.utcnow)
    is_correct = db.Column(db.Boolean)  # User feedback
    
    def __repr__(self):
        return f'<Prediction {self.id}: {self.predicted_label}>'

class ModelPerformance(db.Model):
    __tablename__ = 'model_performance'
    
    id = db.Column(db.Integer, primary_key=True)
    model_version = db.Column(db.String(50), nullable=False)
    accuracy = db.Column(db.Float)
    precision = db.Column(db.Float)
    recall = db.Column(db.Float)
    f1_score = db.Column(db.Float)
    training_data_size = db.Column(db.Integer)
    training_date = db.Column(db.DateTime, default=datetime.utcnow)
    notes = db.Column(db.Text)
    
    def __repr__(self):
        return f'<ModelPerformance {self.model_version}: Acc={self.accuracy:.3f}>'

class SystemLog(db.Model):
    __tablename__ = 'system_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    action = db.Column(db.String(100), nullable=False)
    details = db.Column(db.Text)
    ip_address = db.Column(db.String(45))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<SystemLog {self.action}: {self.timestamp}>'
