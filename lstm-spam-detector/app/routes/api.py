# app/routes/api.py
from flask import Blueprint, jsonify, request
from flask_login import login_required, current_user
from ..models.database import db, Email, Prediction
from ..models.ml_model import EnhancedLSTMSpamDetector

api_bp = Blueprint('api', __name__)

def get_ml_model():
    from flask import current_app
    return EnhancedLSTMSpamDetector(current_app.config)

@api_bp.route('/predict', methods=['POST'])
@login_required
def predict_text():
    """API endpoint for spam prediction"""
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({'error': 'Text is required'}), 400
    
    text = data['text']
    subject = data.get('subject', '')
    sender = data.get('sender', '')
    
    # Save email to database
    email = Email(
        user_id=current_user.id,
        subject=subject,
        content=text,
        sender=sender
    )
    db.session.add(email)
    db.session.commit()
    
    # Make prediction
    model = get_ml_model()
    result = model.predict_email(email.id)
    
    if result:
        return jsonify({
            'success': True,
            'email_id': email.id,
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'probability': result['probability']
        })
    else:
        return jsonify({'error': 'Prediction failed'}), 500

@api_bp.route('/emails/<int:email_id>/feedback', methods=['POST'])
@login_required
def update_feedback(email_id):
    """Update prediction feedback"""
    data = request.get_json()
    
    if not data or 'is_correct' not in data:
        return jsonify({'error': 'Feedback is required'}), 400
    
    prediction = Prediction.query.filter_by(
        email_id=email_id,
        user_id=current_user.id
    ).first()
    
    if not prediction:
        return jsonify({'error': 'Prediction not found'}), 404
    
    prediction.is_correct = data['is_correct']
    db.session.commit()
    
    return jsonify({'success': True})

@api_bp.route('/stats')
@login_required
def get_stats():
    """Get user statistics"""
    total_emails = Email.query.filter_by(user_id=current_user.id).count()
    total_predictions = Prediction.query.filter_by(user_id=current_user.id).count()
    
    spam_count = Prediction.query.filter_by(
        user_id=current_user.id, 
        predicted_label='spam'
    ).count()
    
    return jsonify({
        'total_emails': total_emails,
        'total_predictions': total_predictions,
        'spam_count': spam_count,
        'ham_count': total_predictions - spam_count
    })
