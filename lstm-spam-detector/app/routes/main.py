# app/routes/main.py
from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for
from flask_login import login_required, current_user
from ..models.database import db, Email, Prediction, ModelPerformance, SystemLog
from ..models.ml_model import EnhancedLSTMSpamDetector
from datetime import datetime, timedelta
import os

main_bp = Blueprint('main', __name__)

# Initialize ML model
ml_model = None

def get_ml_model():
    global ml_model
    if ml_model is None:
        from flask import current_app
        ml_model = EnhancedLSTMSpamDetector(current_app.config)
    return ml_model

@main_bp.route('/')
def index():
    return render_template('index.html')

@main_bp.route('/dashboard')
@login_required
def dashboard():
    # Get user statistics
    total_emails = Email.query.filter_by(user_id=current_user.id).count()
    total_predictions = Prediction.query.filter_by(user_id=current_user.id).count()
    
    # Recent predictions
    recent_predictions = db.session.query(Prediction, Email).join(Email).filter(
        Prediction.user_id == current_user.id
    ).order_by(Prediction.prediction_time.desc()).limit(10).all()
    
    # Spam vs Ham distribution
    spam_count = Prediction.query.filter_by(
        user_id=current_user.id, 
        predicted_label='spam'
    ).count()
    ham_count = total_predictions - spam_count
    
    return render_template('dashboard.html',
                         total_emails=total_emails,
                         total_predictions=total_predictions,
                         recent_predictions=recent_predictions,
                         spam_count=spam_count,
                         ham_count=ham_count)

@main_bp.route('/check_email', methods=['GET', 'POST'])
@login_required
def check_email():
    if request.method == 'POST':
        subject = request.form.get('subject', '')
        content = request.form.get('content', '')
        sender = request.form.get('sender', '')
        
        if not content.strip():
            flash('Email content is required!', 'error')
            return redirect(url_for('main.check_email'))
        
        # Save email to database
        email = Email(
            user_id=current_user.id,
            subject=subject,
            content=content,
            sender=sender,
            recipient=current_user.email
        )
        db.session.add(email)
        db.session.commit()
        
        # Make prediction
        model = get_ml_model()
        result = model.predict_email(email.id)
        
        if result:
            flash(f'Prediction: {result["prediction"].upper()} '
                  f'(Confidence: {result["confidence"]*100:.1f}%)', 
                  'success' if result['prediction'] == 'ham' else 'warning')
        else:
            flash('Unable to make prediction. Please try again.', 'error')
        
        # Log activity
        log = SystemLog(
            user_id=current_user.id,
            action='email_check',
            details=f'Checked email: {subject[:50]}',
            ip_address=request.remote_addr
        )
        db.session.add(log)
        db.session.commit()
        
        return redirect(url_for('main.dashboard'))
    
    return render_template('check_email.html')

@main_bp.route('/email_history')
@login_required
def email_history():
    page = request.args.get('page', 1, type=int)
    emails = db.session.query(Email, Prediction).outerjoin(Prediction).filter(
        Email.user_id == current_user.id
    ).order_by(Email.received_date.desc()).paginate(
        page=page, per_page=20, error_out=False
    )
    
    return render_template('email_history.html', emails=emails)

@main_bp.route('/admin')
@login_required
def admin():
    if not current_user.is_admin:
        flash('Access denied!', 'error')
        return redirect(url_for('main.dashboard'))
    
    # System statistics
    total_users = User.query.count()
    total_emails = Email.query.count()
    total_predictions = Prediction.query.count()
    
    # Recent activity
    recent_logs = SystemLog.query.order_by(
        SystemLog.timestamp.desc()
    ).limit(20).all()
    
    # Model performance
    latest_performance = ModelPerformance.query.order_by(
        ModelPerformance.training_date.desc()
    ).first()
    
    return render_template('admin.html',
                         total_users=total_users,
                         total_emails=total_emails,
                         total_predictions=total_predictions,
                         recent_logs=recent_logs,
                         model_performance=latest_performance)

@main_bp.route('/retrain_model', methods=['POST'])
@login_required
def retrain_model():
    if not current_user.is_admin:
        return jsonify({'error': 'Access denied'}), 403
    
    model = get_ml_model()
    success, message = model.train_from_database()
    
    # Log activity
    log = SystemLog(
        user_id=current_user.id,
        action='model_retrain',
        details=message,
        ip_address=request.remote_addr
    )
    db.session.add(log)
    db.session.commit()
    
    return jsonify({
        'success': success,
        'message': message
    })

@main_bp.route('/feedback', methods=['POST'])
@login_required
def feedback():
    prediction_id = request.form.get('prediction_id')
    is_correct = request.form.get('is_correct') == 'true'
    
    prediction = Prediction.query.filter_by(
        id=prediction_id,
        user_id=current_user.id
    ).first()
    
    if prediction:
        prediction.is_correct = is_correct
        db.session.commit()
        flash('Thank you for your feedback!', 'success')
    else:
        flash('Prediction not found!', 'error')
    
    return redirect(url_for('main.email_history'))
