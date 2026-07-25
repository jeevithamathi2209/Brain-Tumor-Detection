"""
app.py - Main Flask Application for Brain Tumor Detection System
Author: Brain Tumor Detection Team
"""

import os
import uuid
import bcrypt
import pandas as pd
from datetime import datetime, timedelta
from functools import wraps
from flask import (
    Flask, render_template, request, redirect, url_for,
    session, flash, jsonify, send_file
)
from werkzeug.utils import secure_filename
from predict import predict_tumor
from report_generator import generate_pdf_report

# ─── App Configuration ────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=2)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['REPORTS_FOLDER'] = 'static/reports'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

# ─── Excel File Paths ─────────────────────────────────────────────────────────
EXCEL_DIR = 'excel_data'
USERS_FILE = os.path.join(EXCEL_DIR, 'users.xlsx')
PREDICTIONS_FILE = os.path.join(EXCEL_DIR, 'predictions.xlsx')

# ─── Ensure Directories Exist ─────────────────────────────────────────────────
for d in [EXCEL_DIR, app.config['UPLOAD_FOLDER'], app.config['REPORTS_FOLDER'],
          'models', 'static/images']:
    os.makedirs(d, exist_ok=True)


# ─── Excel Helpers ────────────────────────────────────────────────────────────
def init_excel_files():
    """Initialize Excel files with headers if they don't exist."""
    if not os.path.exists(USERS_FILE):
        df = pd.DataFrame(columns=[
            'user_id', 'username', 'email', 'password_hash',
            'full_name', 'created_at', 'last_login'
        ])
        df.to_excel(USERS_FILE, index=False)

    if not os.path.exists(PREDICTIONS_FILE):
        df = pd.DataFrame(columns=[
            'prediction_id', 'user_id', 'username', 'image_filename',
            'tumor_class', 'display_name', 'confidence', 'description',
            'is_tumor', 'report_filename', 'created_at'
        ])
        df.to_excel(PREDICTIONS_FILE, index=False)


def read_users() -> pd.DataFrame:
    init_excel_files()
    return pd.read_excel(USERS_FILE, dtype=str)


def write_users(df: pd.DataFrame):
    df.to_excel(USERS_FILE, index=False)


def read_predictions() -> pd.DataFrame:
    init_excel_files()
    try:
        return pd.read_excel(PREDICTIONS_FILE, dtype=str)
    except Exception:
        return pd.DataFrame(columns=[
            'prediction_id', 'user_id', 'username', 'image_filename',
            'tumor_class', 'display_name', 'confidence', 'description',
            'is_tumor', 'report_filename', 'created_at'
        ])


def write_predictions(df: pd.DataFrame):
    df.to_excel(PREDICTIONS_FILE, index=False)


# ─── Utility ──────────────────────────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    return ('.' in filename and
            filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS'])


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


# ─── Initialize ───────────────────────────────────────────────────────────────
init_excel_files()


# ─── Routes: Public ───────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        flash('Thank you for your message! We will get back to you soon.', 'success')
        return redirect(url_for('contact'))
    return render_template('contact.html')


# ─── Routes: Auth ─────────────────────────────────────────────────────────────
@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        full_name = request.form.get('full_name', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')

        # Validation
        if not all([username, email, full_name, password, confirm_password]):
            flash('All fields are required.', 'error')
            return render_template('register.html')

        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return render_template('register.html')

        if len(password) < 6:
            flash('Password must be at least 6 characters.', 'error')
            return render_template('register.html')

        df = read_users()

        if username in df['username'].values:
            flash('Username already taken.', 'error')
            return render_template('register.html')

        if email in df['email'].values:
            flash('Email already registered.', 'error')
            return render_template('register.html')

        # Hash password & store
        hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        new_user = {
            'user_id': str(uuid.uuid4()),
            'username': username,
            'email': email,
            'password_hash': hashed,
            'full_name': full_name,
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'last_login': ''
        }
        df = pd.concat([df, pd.DataFrame([new_user])], ignore_index=True)
        write_users(df)

        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        df = read_users()
        user_row = df[df['username'] == username]

        if user_row.empty:
            flash('Invalid username or password.', 'error')
            return render_template('login.html')

        user = user_row.iloc[0]
        stored_hash = user['password_hash'].encode('utf-8')

        if bcrypt.checkpw(password.encode('utf-8'), stored_hash):
            session.permanent = True
            session['user_id'] = user['user_id']
            session['username'] = user['username']
            session['full_name'] = user['full_name']
            session['email'] = user['email']

            # Update last login
            df.loc[df['username'] == username, 'last_login'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            write_users(df)

            flash(f'Welcome back, {user["full_name"]}!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password.', 'error')

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))


# ─── Routes: Dashboard ────────────────────────────────────────────────────────
@app.route('/dashboard')
@login_required
def dashboard():
    preds = read_predictions()
    user_preds = preds[preds['user_id'] == session['user_id']]
    total = len(user_preds)
    recent = user_preds.tail(5).iloc[::-1].to_dict('records') if total > 0 else []
    tumor_count = len(user_preds[user_preds['is_tumor'] == 'True'])
    return render_template('dashboard.html', total=total, recent=recent, tumor_count=tumor_count)


# ─── Routes: Upload & Predict ─────────────────────────────────────────────────
@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        if 'mri_image' not in request.files:
            flash('No file selected.', 'error')
            return redirect(request.url)

        file = request.files['mri_image']
        if file.filename == '':
            flash('No file selected.', 'error')
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash('Invalid file type. Please upload PNG, JPG, JPEG, BMP, or GIF.', 'error')
            return redirect(request.url)

        # Save uploaded file
        ext = file.filename.rsplit('.', 1)[1].lower()
        unique_name = f"{uuid.uuid4().hex}.{ext}"
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
        file.save(save_path)

        try:
            # Run prediction
            result = predict_tumor(save_path)

            # Generate PDF report
            prediction_id = str(uuid.uuid4())[:8].upper()
            report_filename = f"report_{prediction_id}.pdf"
            report_path = os.path.join(app.config['REPORTS_FOLDER'], report_filename)
            generate_pdf_report(
                report_path=report_path,
                user_name=session['full_name'],
                image_path=save_path,
                result=result,
                prediction_id=prediction_id,
                created_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            )

            # Save prediction to Excel
            preds = read_predictions()
            new_pred = {
                'prediction_id': prediction_id,
                'user_id': session['user_id'],
                'username': session['username'],
                'image_filename': unique_name,
                'tumor_class': result['class'],
                'display_name': result['display_name'],
                'confidence': str(result['confidence']),
                'description': result['description'],
                'is_tumor': str(result['is_tumor']),
                'report_filename': report_filename,
                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            preds = pd.concat([preds, pd.DataFrame([new_pred])], ignore_index=True)
            write_predictions(preds)

            session['last_prediction_id'] = prediction_id
            return redirect(url_for('result', pred_id=prediction_id))

        except FileNotFoundError as e:
            flash(str(e), 'error')
            return redirect(request.url)
        except Exception as e:
            flash(f'Prediction failed: {str(e)}', 'error')
            return redirect(request.url)

    return render_template('upload.html')


@app.route('/result/<pred_id>')
@login_required
def result(pred_id):
    preds = read_predictions()
    row = preds[(preds['prediction_id'] == pred_id) & (preds['user_id'] == session['user_id'])]
    if row.empty:
        flash('Prediction not found.', 'error')
        return redirect(url_for('dashboard'))
    pred = row.iloc[0].to_dict()

    # Build probability dict from stored data (simplified display)
    return render_template('result.html', pred=pred)


# ─── Routes: History ──────────────────────────────────────────────────────────
@app.route('/history')
@login_required
def history():
    preds = read_predictions()
    user_preds = preds[preds['user_id'] == session['user_id']].iloc[::-1]
    records = user_preds.to_dict('records')
    return render_template('history.html', predictions=records)


@app.route('/history/delete/<pred_id>', methods=['POST'])
@login_required
def delete_prediction(pred_id):
    preds = read_predictions()
    row = preds[(preds['prediction_id'] == pred_id) & (preds['user_id'] == session['user_id'])]
    if not row.empty:
        r = row.iloc[0]
        # Delete associated files
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], r['image_filename'])
        rep_path = os.path.join(app.config['REPORTS_FOLDER'], r['report_filename'])
        for p in [img_path, rep_path]:
            if os.path.exists(p):
                os.remove(p)
        preds = preds[~((preds['prediction_id'] == pred_id) & (preds['user_id'] == session['user_id']))]
        write_predictions(preds)
        flash('Prediction record deleted.', 'success')
    else:
        flash('Record not found.', 'error')
    return redirect(url_for('history'))


# ─── Routes: Download Report ──────────────────────────────────────────────────
@app.route('/download_report/<pred_id>')
@login_required
def download_report(pred_id):
    preds = read_predictions()
    row = preds[(preds['prediction_id'] == pred_id) & (preds['user_id'] == session['user_id'])]
    if row.empty:
        flash('Report not found.', 'error')
        return redirect(url_for('history'))
    report_filename = row.iloc[0]['report_filename']
    report_path = os.path.join(app.config['REPORTS_FOLDER'], report_filename)
    if not os.path.exists(report_path):
        flash('Report file not found on server.', 'error')
        return redirect(url_for('history'))
    return send_file(report_path, as_attachment=True, download_name=report_filename)


# ─── Routes: Profile ──────────────────────────────────────────────────────────
@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':
        action = request.form.get('action')
        df = read_users()

        if action == 'update_info':
            full_name = request.form.get('full_name', '').strip()
            email = request.form.get('email', '').strip()
            if not full_name or not email:
                flash('Name and email are required.', 'error')
            else:
                # Check email uniqueness
                others = df[df['user_id'] != session['user_id']]
                if email in others['email'].values:
                    flash('Email already in use.', 'error')
                else:
                    df.loc[df['user_id'] == session['user_id'], 'full_name'] = full_name
                    df.loc[df['user_id'] == session['user_id'], 'email'] = email
                    write_users(df)
                    session['full_name'] = full_name
                    session['email'] = email
                    flash('Profile updated successfully.', 'success')

        elif action == 'change_password':
            current_pw = request.form.get('current_password', '')
            new_pw = request.form.get('new_password', '')
            confirm_pw = request.form.get('confirm_password', '')
            user_row = df[df['user_id'] == session['user_id']].iloc[0]
            if not bcrypt.checkpw(current_pw.encode(), user_row['password_hash'].encode()):
                flash('Current password is incorrect.', 'error')
            elif new_pw != confirm_pw:
                flash('New passwords do not match.', 'error')
            elif len(new_pw) < 6:
                flash('Password must be at least 6 characters.', 'error')
            else:
                hashed = bcrypt.hashpw(new_pw.encode(), bcrypt.gensalt()).decode()
                df.loc[df['user_id'] == session['user_id'], 'password_hash'] = hashed
                write_users(df)
                flash('Password changed successfully.', 'success')

    return render_template('profile.html')


# ─── API: Search Predictions ──────────────────────────────────────────────────
@app.route('/api/search_predictions')
@login_required
def search_predictions():
    query = request.args.get('q', '').lower()
    filter_type = request.args.get('filter', 'all')
    preds = read_predictions()
    user_preds = preds[preds['user_id'] == session['user_id']].iloc[::-1]

    if filter_type == 'tumor':
        user_preds = user_preds[user_preds['is_tumor'] == 'True']
    elif filter_type == 'notumor':
        user_preds = user_preds[user_preds['is_tumor'] == 'False']

    if query:
        mask = (
            user_preds['display_name'].str.lower().str.contains(query, na=False) |
            user_preds['prediction_id'].str.lower().str.contains(query, na=False) |
            user_preds['created_at'].str.lower().str.contains(query, na=False)
        )
        user_preds = user_preds[mask]

    return jsonify(user_preds.to_dict('records'))


# ─── Error Handlers ───────────────────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return render_template('error.html', code=404, message='Page not found.'), 404


@app.errorhandler(413)
def too_large(e):
    flash('File is too large. Maximum size is 16MB.', 'error')
    return redirect(url_for('upload'))


@app.errorhandler(500)
def server_error(e):
    return render_template('error.html', code=500, message='Internal server error.'), 500


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
