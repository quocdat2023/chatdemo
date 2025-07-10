from flask import Flask, request, jsonify, session, render_template, redirect, url_for
from flask_socketio import SocketIO, emit, disconnect  # Add SocketIO
from pymongo import MongoClient
from datetime import datetime, timedelta
from gemini_handler import GeminiHandler, GenerationConfig, Strategy, KeyRotationStrategy
from langchain.memory import ConversationBufferMemory
import json
from typing import List, Dict
import re
import logging
import pickle
import faiss
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from bson import ObjectId
import hashlib
import os
import smtplib
from email.mime.text import MIMEText
import random
import string
from functools import wraps

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Flask app and SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = 'NQJWnj7YwbQML8yE'  # Replace with a secure key
app.config['MONGO_URI'] = 'mongodb+srv://itdatit12:NQJWnj7YwbQML8yE@cluster0.pwv2g0y.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0'
app.config['SMTP_SERVER'] = 'smtp.gmail.com'
app.config['SMTP_PORT'] = 587
app.config['EMAIL_ADDRESS'] = 'legalmind2025@gmail.com'  # Replace with your email
app.config['EMAIL_PASSWORD'] = 'hihj vpcb ayjk gaex'  # Replace with your app password
socketio = SocketIO(app, cors_allowed_origins="*")  # Initialize SocketIO

# Initialize MongoDB client
mongo = MongoClient(app.config['MONGO_URI'])
db = mongo.get_database('legal_assistant')

# Store WebSocket clients by user_id
connected_clients = {}

# Password hashing function
def hash_password(password: str) -> str:
    salt = os.urandom(32)
    hashed = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
    password_hash = (salt + hashed).hex()
    logging.info(f"Generated password hash: {password_hash}")
    return password_hash

# Password verification function
def verify_password(stored_password: str, provided_password: str) -> bool:
    if not stored_password or not all(c in '0123456789abcdefABCDEF' for c in stored_password):
        logging.error(f"Invalid stored password format: {stored_password}")
        return False
    try:
        stored_bytes = bytes.fromhex(stored_password)
        salt = stored_bytes[:32]
        stored_hash = stored_bytes[32:]
        provided_hash = hashlib.pbkdf2_hmac('sha256', provided_password.encode('utf-8'), salt, 100000)
        return stored_hash == provided_hash
    except ValueError as e:
        logging.error(f"Error in verify_password: {e}, stored_password: {stored_password}")
        return False

# Generate OTP
def generate_otp(length=6):
    return ''.join(random.choices(string.digits, k=length))

# Send email with OTP or password
def send_email(to_email, subject, body):
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = app.config['EMAIL_ADDRESS']
    msg['To'] = to_email
    try:
        with smtplib.SMTP(app.config['SMTP_SERVER'], app.config['SMTP_PORT']) as server:
            server.starttls()
            server.login(app.config['EMAIL_ADDRESS'], app.config['EMAIL_PASSWORD'])
            server.send_message(msg)
        logging.info(f"Email sent to {to_email}")
        return True
    except Exception as e:
        logging.error(f"Error sending email: {e}")
        return False

# Khởi tạo model embedding
model = SentenceTransformer('hiieu/halong_embedding', device='cuda' if torch.cuda.is_available() else 'cpu')

# Đường dẫn đến FAISS index và embeddings data
INDEX_PATH = "embedding_data/faiss_index_23_06.index"
EMBEDDINGS_DATA_PATH = "embedding_data/embeddings.pkl"

# Khởi tạo ConversationBufferMemory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    max_message_limit=10,
    max_token_limit=1000
)

# Load FAISS index
def load_faiss_index(index_path):
    try:
        index = faiss.read_index(index_path)
        logging.info(f"Loaded FAISS index from {index_path}")
        return index
    except Exception as e:
        logging.error(f"Error loading FAISS index: {e}")
        return None

# Load embeddings data
def load_embeddings_data(data_path):
    try:
        with open(data_path, 'rb') as f:
            embeddings_data = pickle.load(f)
        logging.info(f"Loaded embeddings data from {data_path}")
        return embeddings_data
    except Exception as e:
        logging.error(f"Error loading embeddings data: {e}")
        return None

# Hàm truy xuất
def retrieve(query, index, embeddings_data, k=10):
    try:
        query_embedding = model.encode([query], convert_to_numpy=True)
        distances, indices = index.search(query_embedding, k)
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            results.append({
                'file': embeddings_data[idx]['file'],
                'folder': embeddings_data[idx]['folder'],
                'text_path': embeddings_data[idx]['text_path'],
                'text': embeddings_data[idx]['text'],
                'distance': float(distance)
            })
        return results
    except Exception as e:
        logging.error(f"Error during retrieval: {e}")
        return []

# Load FAISS index và embeddings data
index = load_faiss_index(INDEX_PATH)
embeddings_data = load_embeddings_data(EMBEDDINGS_DATA_PATH)
if index is None or embeddings_data is None:
    logging.error("Failed to load FAISS index or embeddings data. Application cannot start.")
    exit(1)

# Reset query count for limited accounts (daily reset)
def reset_query_count(user_id):
    user = db.users.find_one({'_id': ObjectId(user_id)})
    if not user or user.get('account_type') == 'unlimited':
        return
    last_reset = user.get('last_reset')
    if last_reset and datetime.utcnow() - last_reset > timedelta(days=1):
        db.users.update_one(
            {'_id': ObjectId(user_id)},
            {'$set': {'query_count': 0, 'last_reset': datetime.utcnow()}}
        )
        logging.info(f"Query count reset for user {user_id}")

# Check if user can make a query
def can_make_query(user_id):
    user = db.users.find_one({'_id': ObjectId(user_id)})
    if not user:
        return False, "Người dùng không tồn tại", None, None
    if user.get('is_admin') or user.get('account_type') == 'unlimited':
        return True, None, None, None
    reset_query_count(user_id)
    user = db.users.find_one({'_id': ObjectId(user_id)})
    query_limit = user.get('query_limit', 10)
    query_count = user.get('query_count', 0)
    if query_count >= query_limit:
        return False, f"Bạn đã sử dụng hết {query_limit} lượt hỏi đáp hôm nay", query_count, query_limit
    # Warn if user is close to the limit
    if query_count + 1 == query_limit:
        return True, "Cảnh báo: Đây là lượt hỏi cuối cùng của bạn hôm nay", query_count, query_limit
    return True, None, query_count, query_limit

# Admin required decorator
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Vui lòng đăng nhập'}), 401
        user = db.users.find_one({'_id': ObjectId(session['user_id'])})
        if not user or not user.get('is_admin'):
            return jsonify({'error': 'Quyền truy cập bị từ chối. Chỉ admin được phép.'}), 403
        return f(*args, **kwargs)
    return decorated_function

# Preprocess related questions
def preprocess_related_questions(related_questions_input: str | List[Dict[str, str]]) -> List[Dict[str, str]]:
    fallback_questions = [
        {"question": "Quy định pháp luật Việt Nam hiện hành về xử lý tranh chấp hợp đồng dân sự được quy định trong văn bản nào?"},
        {"question": "Trường hợp nào thì một bản án có thể được sử dụng làm án lệ theo quy định của pháp luật Việt Nam?"},
        {"question": "Các nguyên tắc cơ bản của Bộ luật Dân sự Việt Nam năm 2015 được quy định tại điều khoản nào?"},
        {"question": "Nghị định nào quy định về xử phạt vi phạm hành chính trong lĩnh vực hôn nhân và gia đình tại Việt Nam?"},
        {"question": "Quy trình áp dụng pháp luật trong trường hợp không có bản án tương đồng được thực hiện như thế nào?"}
    ]
    if isinstance(related_questions_input, str):
        cleaned_input = re.sub(r'^```json\s*|\s*```$', '', related_questions_input).strip()
        try:
            related_questions = json.loads(cleaned_input)
        except json.JSONDecodeError:
            return fallback_questions[:5]
    else:
        related_questions = related_questions_input
    if not isinstance(related_questions, list):
        return fallback_questions[:5]
    valid_questions = [
        q for q in related_questions
        if isinstance(q, dict) and "question" in q and isinstance(q["question"], str) and q["question"].strip()
    ]
    seen = set()
    unique_questions = []
    for q in valid_questions:
        question_text = q["question"].strip()
        if question_text not in seen:
            seen.add(question_text)
            unique_questions.append({"question": question_text})
    legal_keywords = r"(Luật|Bộ luật|Nghị định|Thông tư|Quy định|án lệ|Việt Nam|tòa án|pháp luật|điều luật|Bảo hiểm xã hội)"
    filtered_questions = [
        q for q in unique_questions
        if re.search(legal_keywords, q["question"], re.IGNORECASE)
    ]
    if len(filtered_questions) < 5:
        remaining = 5 - len(filtered_questions)
        for fq in fallback_questions:
            if len(filtered_questions) >= 5:
                break
            if fq["question"] not in seen:
                filtered_questions.append(fq)
                seen.add(fq["question"])
    return filtered_questions[:5]

def format_chat_history(memory):
    messages = memory.chat_memory.messages
    if not messages:
        return "Không có lịch sử hội thoại trước."
    formatted = []
    for m in messages:
        role = getattr(m, "type", None) or m.get("role", "User")
        content = getattr(m, "content", None) or m.get("content", "")
        formatted.append(f"{role.capitalize()}: {content}")
    return "\n".join(formatted)

# WebSocket handlers
@socketio.on('connect')
def handle_connect():
    user_id = session.get('user_id')
    if user_id:
        connected_clients[user_id] = request.sid
        logging.info(f"User {user_id} connected via WebSocket with SID {request.sid}")
    else:
        disconnect()  # Disconnect unauthorized clients
        logging.warning("Unauthorized WebSocket connection attempt")

@socketio.on('disconnect')
def handle_disconnect():
    user_id = session.get('user_id')
    if user_id in connected_clients and connected_clients[user_id] == request.sid:
        del connected_clients[user_id]
        logging.info(f"User {user_id} disconnected from WebSocket")

# Đăng ký
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return render_template('register.html')
    data = request.get_json(silent=True) or {}
    username = data.get('username', '').strip()
    email = data.get('email', '').strip()
    password = data.get('password', '').strip()
    phone = data.get('phone', '').strip()
    account_type = data.get('account_type', 'limited').strip()
    if not username or not email or not password or not phone:
        return jsonify({'error': 'Thiếu thông tin bắt buộc'}), 400
    if not re.match(r'^\+84\d{9}$|^0\d{9}$', phone):
        return jsonify({'error': 'Số điện thoại không hợp lệ'}), 400
    if account_type not in ['limited', 'unlimited']:
        return jsonify({'error': 'Loại tài khoản không hợp lệ'}), 400
    if db.users.find_one({'$or': [{'email': email}, {'phone': phone}]}):
        return jsonify({'error': 'Email hoặc số điện thoại đã tồn tại'}), 400
    otp = generate_otp()
    password_hash = hash_password(password)
    user = {
        'username': username,
        'email': email,
        'phone': phone,
        'password_hash': password_hash,
        'otp': otp,
        'is_active': False,
        'created_at': datetime.utcnow(),
        'is_admin': False,
        'account_type': account_type,
        'query_limit': 3 if account_type == 'limited' else None,
        'query_count': 0,
        'last_reset': datetime.utcnow()
    }
    result = db.users.insert_one(user)
    if send_email(
        email,
        'Mã OTP xác thực tài khoản',
        f'Mã OTP của bạn là: {otp}. Vui lòng sử dụng mã này để xác thực tài khoản.'
    ):
        return jsonify({
            'message': 'Đăng ký thành công, vui lòng kiểm tra email để lấy mã OTP',
            'user_id': str(result.inserted_id)
        }), 201
    else:
        db.users.delete_one({'_id': result.inserted_id})
        return jsonify({'error': 'Lỗi khi gửi OTP, vui lòng thử lại'}), 500

# Xác thực OTP
@app.route('/verify_otp', methods=['GET', 'POST'])
def verify_otp():
    if request.method == 'GET':
        user_id = request.args.get('user_id')
        if not user_id:
            return jsonify({'error': 'Thiếu user_id'}), 400
        try:
            user = db.users.find_one({'_id': ObjectId(user_id)})
            if not user:
                return jsonify({'error': 'Người dùng không tồn tại'}), 404
            return render_template('verify_otp.html', user_id=user_id)
        except Exception as e:
            logging.error(f"Invalid user_id: {e}")
            return jsonify({'error': 'user_id không hợp lệ'}), 400
    elif request.method == 'POST':
        data = request.get_json(silent=True) or {}
        user_id = data.get('user_id', '').strip()
        otp = data.get('otp', '').strip()
        if not user_id or not otp:
            return jsonify({'error': 'Thiếu user_id hoặc OTP'}), 400
        try:
            user = db.users.find_one({'_id': ObjectId(user_id)})
            if not user:
                return jsonify({'error': 'Người dùng không tồn tại'}), 404
            if user.get('otp') != otp:
                return jsonify({'error': 'Mã OTP không đúng'}), 400
            db.users.update_one(
                {'_id': ObjectId(user_id)},
                {'$set': {'is_active': True, 'otp': None}}
            )
            return jsonify({'message': 'Xác thực tài khoản thành công'}), 200
        except Exception as e:
            logging.error(f"Error verifying OTP: {e}")
            return jsonify({'error': 'Lỗi hệ thống, vui lòng thử lại'}), 500

# Get masked phone number
@app.route('/get_masked_phone', methods=['POST'])
def get_masked_phone():
    data = request.get_json(silent=True) or {}
    email = data.get('email', '').strip()
    if not email:
        return jsonify({'error': 'Thiếu email'}), 400
    user = db.users.find_one({'email': email})
    if not user:
        return jsonify({'error': 'Email không tồn tại'}), 404
    phone = user.get('phone', '')
    masked_phone = phone[:-4] + '****'
    return jsonify({'masked_phone': masked_phone}), 200

# Quên mật khẩu
@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'GET':
        return render_template('forgot_password.html')
    elif request.method == 'POST':
        data = request.get_json(silent=True) or {}
        email = data.get('email', '').strip()
        last_four_digits = data.get('last_four_digits', '').strip()
        if not email or not last_four_digits:
            return jsonify({'error': 'Thiếu email hoặc 4 số cuối của số điện thoại'}), 400
        user = db.users.find_one({'email': email})
        if not user:
            return jsonify({'error': 'Email không tồn tại'}), 404
        phone = user.get('phone', '')
        if not phone[-4:] == last_four_digits:
            return jsonify({'error': '4 số cuối của số điện thoại không khớp'}), 400
        new_password = ''.join(random.choices(string.ascii_letters + string.digits, k=12))
        new_password_hash = hash_password(new_password)
        db.users.update_one(
            {'_id': user['_id']},
            {'$set': {'password_hash': new_password_hash}}
        )
        if send_email(
            email,
            'Mật khẩu mới',
            f'Mật khẩu mới của bạn là: {new_password}. Vui lòng đổi mật khẩu sau khi đăng nhập.'
        ):
            return jsonify({'message': 'Mật khẩu mới đã được gửi qua email'}), 200
        else:
            return jsonify({'error': 'Lỗi khi gửi mật khẩu mới'}), 500

@app.route('/change_password', methods=['GET'])
def change_password_get():
    if 'user_id' not in session:
        return redirect(url_for('login_page'))
    return render_template('change_password.html')

# Đổi mật khẩu
@app.route('/change_password', methods=['POST'])
def change_password():
    if 'user_id' not in session:
        return jsonify({'error': 'Vui lòng đăng nhập để đổi mật khẩu'}), 401

    data = request.get_json(silent=True) or {}
    current_password = data.get('current_password', '').strip()
    new_password = data.get('new_password', '').strip()

    if not current_password or not new_password:
        return jsonify({'error': 'Thiếu mật khẩu hiện tại hoặc mật khẩu mới'}), 400

    user_id = session['user_id']
    user = db.users.find_one({'_id': ObjectId(user_id)})
    if not user:
        return jsonify({'error': 'Người dùng không tồn tại'}), 404

    if not verify_password(user['password_hash'], current_password):
        return jsonify({'error': 'Mật khẩu hiện tại không đúng'}), 401

    new_password_hash = hash_password(new_password)
    db.users.update_one(
        {'_id': ObjectId(user_id)},
        {'$set': {'password_hash': new_password_hash}}
    )

    # Send confirmation email
    if send_email(
        user['email'],
        'Xác nhận đổi mật khẩu',
        f'Mật khẩu của bạn đã được thay đổi thành công vào lúc {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")}.'
    ):
        return jsonify({'message': 'Đổi mật khẩu thành công, email xác nhận đã được gửi'}), 200
    else:
        return jsonify({'error': 'Đổi mật khẩu thành công nhưng lỗi khi gửi email xác nhận'}), 200

# # Đăng nhập
@app.route('/logins', methods=['POST'])
def login():
    data = request.get_json(silent=True) or {}
    email = data.get('email', '').strip()
    password = data.get('password', '').strip()
    user = db.users.find_one({'email': email})
    if not user:
        logging.error(f"No user found for email: {email}")
        return jsonify({'error': 'Email hoặc mật khẩu không đúng'}), 401
    if not user.get('is_active', False):
        return jsonify({'error': 'Tài khoản chưa được kích hoạt. Vui lòng xác thực OTP.'}), 401
    if not verify_password(user['password_hash'], password):
        return jsonify({'error': 'Email hoặc mật khẩu không đúng'}), 401
    session['user_id'] = str(user['_id'])
    session['username'] = user['username']
    session['is_admin'] = user.get('is_admin', False)
    session['query_limit'] = user.get('query_limit', 3)
    session['query_count'] = user.get('query_count', 0)
    session['account_type'] = user.get('account_type', 'limited')
    
    # Broadcast initial query count and limit
    if str(user['_id']) in connected_clients:
        socketio.emit('query_update', {
            'query_count': user.get('query_count', 0),
            'query_limit': user.get('query_limit', 3 if user.get('account_type') == 'limited' else None)
        }, room=connected_clients[str(user['_id'])])
        logging.info(f"Broadcasted initial query update to user {user['_id']}")
    

    return jsonify({
        'message': 'Đăng nhập thành công',
        'username': user['username'],
        'is_admin': user.get('is_admin', False),
        'account_type': user.get('account_type', 'limited'),
        'query_limit': user.get('query_limit', 3),
        'query_count': user.get('query_count', 0),
    }), 200


# @app.route('/logins', methods=['POST'])
# def login():
#     data = request.get_json(silent=True) or {}
#     email = data.get('email', '').strip()
#     password = data.get('password', '').strip()
#     user = db.users.find_one({'email': email})
#     if not user:
#         logging.error(f"No user found for email: {email}")
#         return jsonify({'error': 'Email hoặc mật khẩu không đúng'}), 401
#     if not user.get('is_active', False):
#         return jsonify({'error': 'Tài khoản chưa được kích hoạt. Vui lòng xác thực OTP.'}), 401
#     if not verify_password(user['password_hash'], password):
#         return jsonify({'error': 'Email hoặc mật khẩu không đúng'}), 401
#     session['user_id'] = str(user['_id'])
#     session['username'] = user['username']
#     session['is_admin'] = user.get('is_admin', False)
#     session['query_limit'] = user.get('query_limit', 3)
#     session['query_count'] = user.get('query_count', 0)
#     session['account_type'] = user.get('account_type', 'limited')
#     return jsonify({
#         'message': 'Đăng nhập thành công',
#         'username': user['username'],
#         'is_admin': user.get('is_admin', False),
#         'account_type': user.get('account_type', 'limited'),
#         'query_limit': user.get('query_limit', 3),
#         'query_count': user.get('query_count', 0)   
#     }), 200

# Đăng xuất
@app.route('/logout', methods=['POST'])
def logout():
    user_id = session.get('user_id')
    if user_id in connected_clients:
        del connected_clients[user_id]  # Remove from connected clients
        logging.info(f"User {user_id} removed from connected clients on logout")
    session.pop('user_id', None)
    session.pop('username', None)
    session.pop('is_admin', None)
    session.pop('account_type', None)
    return jsonify({'message': 'Đăng xuất thành công'}), 200

# Kiểm tra session
@app.route('/check_session', methods=['GET'])
def check_session():
    if 'user_id' in session:
        user = db.users.find_one({'_id': ObjectId(session['user_id'])})
        if user:
            # Update session with latest values
            session['query_limit'] = user.get('query_limit', 3)
            session['query_count'] = user.get('query_count', 0)
            session['account_type'] = user.get('account_type', 'limited')
            # Broadcast current query count and limit
            if session['user_id'] in connected_clients:
                socketio.emit('query_update', {
                    'query_count': user.get('query_count', 0),
                    'query_limit': user.get('query_limit', 3 if user.get('account_type') == 'limited' else None)
                }, room=connected_clients[session['user_id']])
                logging.info(f"Broadcasted query update to user {session['user_id']} on session check")
            return jsonify({
                'logged_in': True,
                'username': session['username'],
                'query_limit': session['query_limit'],
                'query_count': session['query_count'],
                'is_admin': session.get('is_admin', False),
                'account_type': session['account_type']
            }), 200
    return jsonify({'logged_in': False}), 200
# @app.route('/check_session', methods=['GET'])
# def check_session():
#     if 'user_id' in session:
#         return jsonify({
#             'logged_in': True,
#             'username': session['username'],
#             'query_limit': session.get('query_limit', 3),
#             'query_count': session.get('query_count', 0),
#             'is_admin': session.get('is_admin', False),
#             'account_type': session.get('account_type', 'limited')
#         }), 200
#     return jsonify({'logged_in': False}), 200


@socketio.on('connect')
def handle_connect():
    user_id = session.get('user_id')
    if user_id:
        connected_clients[user_id] = request.sid
        user = db.users.find_one({'_id': ObjectId(user_id)})
        if user:
            socketio.emit('query_update', {
                'query_count': user.get('query_count', 0),
                'query_limit': user.get('query_limit', 3 if user.get('account_type') == 'limited' else None)
            }, room=request.sid)
            logging.info(f"Emitted initial query_update to user {user_id} on connect")

# Lấy danh sách hội thoại
@app.route('/conversations', methods=['GET'])
def get_conversations():
    if 'user_id' not in session:
        return jsonify({'error': 'Vui lòng đăng nhập'}), 401
    user_id = session['user_id']
    conversations = db.conversations.find({'user_id': user_id}).sort('timestamp', -1)
    return jsonify([{
        'id': str(conv['_id']),
        'title': conv['title'],
        'timestamp': conv['timestamp'].isoformat(),
        'message_count': db.messages.count_documents({'conversation_id': str(conv['_id'])})
    } for conv in conversations])

# Lấy chi tiết hội thoại
@app.route('/conversation/<conversation_id>', methods=['GET'])
def get_conversation(conversation_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Vui lòng đăng nhập'}), 401
    user_id = session['user_id']
    conversation = db.conversations.find_one({'_id': ObjectId(conversation_id), 'user_id': user_id})
    if not conversation:
        return jsonify({'error': 'Hội thoại không tồn tại'}), 404
    messages = db.messages.find({'conversation_id': conversation_id})
    messages_list = [{
        'id': str(msg['_id']),
        'type': msg['type'],
        'content': msg['content'],
        'timestamp': msg['timestamp'].isoformat(),
        'sources': msg.get('sources'),
        'related_questions': msg.get('related_questions')
    } for msg in messages]
    return jsonify({
        'id': str(conversation['_id']),
        'title': conversation['title'],
        'timestamp': conversation['timestamp'].isoformat(),
        'messages': messages_list
    })

# Xử lý truy vấn
@app.route("/query", methods=["POST"])
def query():
    if 'user_id' not in session:
        return jsonify({
            'error': 'Vui lòng đăng nhập để sử dụng tính năng này',
            'error_code': 'UNAUTHENTICATED'
        }), 401
    
    user_id = session['user_id']
    
    # Validate user_id format
    try:
        ObjectId(user_id)
    except Exception:
        return jsonify({
            'error': 'ID người dùng không hợp lệ',
            'error_code': 'INVALID_USER_ID'
        }), 400

    # Check query permission
    can_query, error_message, query_count, query_limit = can_make_query(user_id)
    if not can_query:
        return jsonify({
            'error': error_message,  # e.g., "Bạn đã sử dụng hết 10 lượt hỏi đáp hôm nay"
            'error_code': 'QUERY_LIMIT_EXCEEDED',
            'query_count': query_count,
            'query_limit': query_limit,
            'upgrade_url': 'https://legalmindver1.loca.lt'  # Redirect to upgrade page
        }), 403
    elif error_message:  # Warning for last query
        logging.info(f"User {user_id} received warning: {error_message}")

    # Parse JSON input
    data = request.get_json(silent=True) or {}
    question = data.get('question', '').strip()
    if not question:
        return jsonify({
            'error': 'Câu hỏi không hợp lệ',
            'error_code': 'INVALID_QUESTION'
        }), 400

    # Update query count for limited accounts
    user = db.users.find_one({'_id': ObjectId(user_id)})
    if not user:
        return jsonify({
            'error': 'Người dùng không tồn tại',
            'error_code': 'USER_NOT_FOUND'
        }), 404

    if user.get('account_type') == 'limited':
        try:
            db.users.update_one(
                {'_id': ObjectId(user_id)},
                {'$inc': {'query_count': 1}}
            )
            # Fetch updated user data
            user = db.users.find_one({'_id': ObjectId(user_id)})
            query_count = user.get('query_count', 0)
            query_limit = user.get('query_limit', 10)
            user_type = user.get('account_type', 'limited')
            # Broadcast updated query count to the user
            if user_id in connected_clients:
                socketio.emit('query_update', {
                    'query_count': query_count,
                    'query_limit': query_limit,
                    'user_type': user_type      
                }, room=connected_clients[user_id])
                logging.info(f"Broadcasted query update to user {user_id}: {query_count}/{query_limit}")
        except Exception as e:
            logging.error(f"Error updating query count for user {user_id}: {e}")
            return jsonify({
                'error': f"Bạn đã sử dụng hết {query_limit} lượt hỏi đáp hôm nay",  # Use specific message
                'error_code': 'DATABASE_ERROR',
                'query_count': query_count,
                'query_limit': query_limit,
                'user_type': user_type,
                'upgrade_url': 'https://legalmindver1.loca.lt/'
            }), 500

    # Handle conversation
    conversation_id = data.get('conversation_id')
    if conversation_id:
        try:
            conversation = db.conversations.find_one({'_id': ObjectId(conversation_id), 'user_id': user_id})
            if not conversation:
                return jsonify({
                    'error': 'Hội thoại không tồn tại hoặc không thuộc về người dùng',
                    'error_code': 'CONVERSATION_NOT_FOUND'
                }), 404
        except Exception:
            return jsonify({
                'error': 'ID hội thoại không hợp lệ',
                'error_code': 'INVALID_CONVERSATION_ID'
            }), 400
    else:
        conversation = {
            'user_id': user_id,
            'title': question[:50],
            'timestamp': datetime.utcnow(),
            'messages': []
        }
        try:
            result = db.conversations.insert_one(conversation)
            conversation_id = str(result.inserted_id)
        except Exception as e:
            logging.error(f"Error creating conversation for user {user_id}: {e}")
            return jsonify({
                'error': f"Bạn đã sử dụng hết {query_limit} lượt hỏi đáp hôm nay",  # Use specific message
                'error_code': 'DATABASE_ERROR',
                'query_count': query_count,
                'query_limit': query_limit,
                'upgrade_url': 'https://legalmindver1.loca.lt/'
            }), 500

    # Save user message
    user_message = {
        'conversation_id': conversation_id,
        'type': 'user',
        'content': question,
        'timestamp': datetime.utcnow()
    }
    try:
        db.messages.insert_one(user_message)
    except Exception as e:
        logging.error(f"Error saving user message for conversation {conversation_id}: {e}")
        return jsonify({
            'error': f"Bạn đã sử dụng hết {query_limit} lượt hỏi đáp hôm nay",  # Use specific message
            'error_code': 'DATABASE_ERROR',
            'query_count': query_count,
            'query_limit': query_limit,
            'upgrade_url': 'https://legalmindver1.loca.lt/'
        }), 500

    # Retrieve relevant legal documents
    try:
        banan_results = retrieve(question, index, embeddings_data, k=5)
    except Exception as e:
        logging.error(f"Error retrieving documents: {e}")
        banan_results = []

    # Format chat history
    chat_history_str = format_chat_history(memory)

    # Define main prompt
    main_prompt = f"""
Dưới đây là lịch sử hội thoại trước đó:
{chat_history_str}

**Câu hỏi:**  
{question}

**Thông tin tham khảo (bản án tương đồng):**  
{banan_results if banan_results else "Không tìm thấy bản án phù hợp. Phân tích dựa trên các quy định pháp luật hiện hành và nguyên tắc pháp lý chung."}

**Hướng dẫn trả lời chi tiết:**
1. **Tổng quan về bản án, án lệ tương đồng:**  
   - Trình bày rõ thông tin tham khảo nếu có.
   - Nhớ đề cập đến tên file bản án, không dùng từ ví dụ, giả sử, giả định.
   - Nếu không có bản án, nêu rõ sẽ phân tích trên cơ sở các điều luật hiện hành tại Việt Nam.

2. **Nội dung chi tiết của bản án, án lệ:**  
   - Nếu có thông tin cụ thể, trình bày rõ vấn đề pháp lý và lập luận của tòa án trong bản án, án lệ liên quan.
   - Nếu không có thông tin đầy đủ, phân tích dựa vào các nguyên tắc pháp lý chung, điều luật, nghị định hiện hành.

3. **Phân tích tình huống pháp lý:**  
   - Phân tích rõ các vấn đề pháp lý chính.
   - Làm nổi bật các quy định cụ thể trong Bộ luật Dân sự, Luật Thương mại hoặc các luật chuyên ngành, nghị định, nghị quyết và các văn bản pháp luật liên quan.

4. **Lập luận pháp lý:**  
   - Nêu rõ căn cứ pháp lý chính xác, trích dẫn cụ thể các điều khoản, nghị định, nghị quyết, thông tư, văn bản hướng dẫn thi hành liên quan.
   - Giải thích rõ cách thức áp dụng các điều khoản pháp luật vào tình huống thực tế, bảo đảm chính xác và khả thi trong thực tiễn.

5. **Kết luận và khuyến nghị:**  
   - Kết luận rõ quyền và nghĩa vụ các bên theo quy định của pháp luật.
   - Chỉ ra những hậu quả pháp lý cụ thể, kèm theo lưu ý khi áp dụng vào các tình huống tương tự trong thực tế.

6. **Nguồn tham khảo:**
   - Nguồn trích dẫn pháp luật bao gồm các điều luật, nghị định, nghị quyết, thông tư, văn bản hướng dẫn thi hành liên quan đến vụ án nằm bên trong button <a href="">Tên nội dung tham khảo như là Khoản, điều, luật, nghị định...</a>.
   **Ví dụ:**
   <a href="">[Luật Hôn nhân và Gia đình 2014, số 52/2014/QH13]</a> (Đặc biệt Điều 3, Điều 5, Điều 8, Điều 10, Điều 11, Điều 12)
   <a href="">[Nghị định 115/2015/NĐ-CP]</a> (Đặc biệt Điều 58)
   <a href="">[Bộ luật Hình sự năm 2015]</a> (Đặc biệt Điều 184)

**Lưu ý quan trọng:**  
- Có trả về nguồn trích dẫn điều luật, nghị định, nghị quyết, thông tư, văn bản hướng dẫn thi hành liên quan đến vụ án, dưới dạng <a>Tên nội dung tham khảo</a>.
- Nếu câu hỏi không thuộc lĩnh vực pháp lý hoặc không có thông tin pháp lý phù hợp, hãy trả lời: "Câu trả lời không nằm trong kiến thức của tôi."
- Trả lời ngắn gọn, súc tích, rõ ràng, đúng trọng tâm.
- Tuyệt đối không dùng từ "giả sử", "ví dụ".
- Không giới thiệu bản thân, không đề cập đến kinh nghiệm tư vấn.
- Không cần mô tả quy trình phân tích.
- Nếu không có thông tin bản án, án lệ phù hợp, hãy bỏ qua, tập trung hoàn toàn vào phân tích pháp luật hiện hành.
- Phân tích phải luôn kết hợp chặt chẽ giữa lý thuyết pháp lý và văn bản pháp luật Việt Nam hiện hành.
- Trình bày rõ ràng, ngắn gọn, sử dụng ngôn ngữ pháp lý chuẩn xác, dễ áp dụng vào thực tế.
"""

    # Call Gemini for main response
    try:
        handler = GeminiHandler(
            config_path="config.yaml",
            content_strategy=Strategy.ROUND_ROBIN,
            key_strategy=KeyRotationStrategy.SMART_COOLDOWN
        )
        gen = handler.generate_content(
            prompt=main_prompt,
            model_name="gemini-2.0-flash-thinking-exp-01-21",
            return_stats=False
        )
        answer = gen.get("text", "Không có phản hồi từ mô hình.")
    except Exception as e:
        logging.error(f"Error calling Gemini for main prompt: {e}")
        return jsonify({
            'error': f"Bạn đã sử dụng hết {query_limit} lượt hỏi đáp hôm nay",  # Use specific message
            'error_code': 'GEMINI_ERROR',
            'query_count': query_count,
            'query_limit': query_limit,
            'upgrade_url': 'https://legalmindver1.loca.lt/'
        }), 500

    # Define related questions prompt
    related_questions_prompt = f"""
Bạn là chuyên gia tư vấn pháp luật Việt Nam. Dựa trên câu hỏi pháp lý được cung cấp, hãy sinh ra 5 câu hỏi liên quan, đảm bảo các câu hỏi:
- Liên quan chặt chẽ đến chủ đề pháp lý của câu hỏi gốc.
- Phù hợp với hệ thống pháp luật Việt Nam hiện hành.
- Ngắn gọn, rõ ràng, và mang tính ứng dụng thực tế.
- Tập trung vào các khía cạnh pháp lý như quy định, điều luật, nghị định, án lệ, hoặc thủ tục pháp lý.
- Được trình bày dưới dạng danh sách JSON, mỗi câu hỏi là một đối tượng với key `question`.

**Câu hỏi gốc:**  
{question}

**Hướng dẫn thêm:**
- Nếu câu hỏi gốc thuộc một lĩnh vực pháp lý cụ thể (ví dụ: dân sự, hình sự, thương mại, hôn nhân và gia đình), hãy sinh ra các câu hỏi liên quan đến lĩnh vực đó.
- Nếu câu hỏi không rõ lĩnh vực, sinh ra các câu hỏi liên quan đến các khía cạnh pháp lý chung như Bộ luật Dân sự, Bộ luật Hình sự, hoặc các nghị định liên quan.
- Không sử dụng từ "giả sử" hoặc "ví dụ".
- Không lặp lại câu hỏi gốc.
- Đảm bảo các câu hỏi có tính liên quan và không trùng lặp nội dung.

**Định dạng đầu ra (JSON):**  
[
  {{"question": "Câu hỏi 1"}},
  {{"question": "Câu hỏi 2"}},
  {{"question": "Câu hỏi 3"}},
  {{"question": "Câu hỏi 4"}},
  {{"question": "Câu hỏi 5"}}
]
"""
    try:
        handler = GeminiHandler(
            config_path="config.yaml",
            content_strategy=Strategy.ROUND_ROBIN,
            key_strategy=KeyRotationStrategy.SMART_COOLDOWN
        )
        gen = handler.generate_content(
            prompt=related_questions_prompt,
            model_name="gemini-2.0-flash-thinking-exp-01-21",
            return_stats=False
        )
        related_questions = gen.get("text", "Không có phản hồi từ mô hình.")
    except Exception as e:
        logging.error(f"Error calling Gemini for related questions: {e}")
        return jsonify({
            'error': f"Bạn đã sử dụng hết {query_limit} lượt hỏi đáp hôm nay",  # Use specific message
            'error_code': 'GEMINI_ERROR',
            'query_count': query_count,
            'query_limit': query_limit,
            'upgrade_url': 'https://legalmindver1.loca.lt/'
        }), 500

    related_questions = preprocess_related_questions(related_questions)
    assistant_message = {
        'conversation_id': conversation_id,
        'type': 'assistant',
        'content': answer,
        'timestamp': datetime.utcnow(),
        'sources': banan_results,
        'related_questions': related_questions
    }
    try:
        db.messages.insert_one(assistant_message)
    except Exception as e:
        logging.error(f"Error saving assistant message for conversation {conversation_id}: {e}")
        return jsonify({
            'error': f"Bạn đã sử dụng hết {query_limit} lượt hỏi đáp hôm nay",  # Use specific message
            'error_code': 'DATABASE_ERROR',
            'query_count': query_count,
            'query_limit': query_limit,
            'upgrade_url': 'https://legalmindver1.loca.lt/'
        }), 500

    try:
        db.conversations.update_one(
            {'_id': ObjectId(conversation_id)},
            {'$set': {'title': question[:50], 'timestamp': datetime.utcnow()}}
        )
    except Exception as e:
        logging.error(f"Error updating conversation {conversation_id}: {e}")
        return jsonify({
            'error': f"Bạn đã sử dụng hết {query_limit} lượt hỏi đáp hôm nay",  # Use specific message
            'error_code': 'DATABASE_ERROR',
            'query_count': query_count,
            'query_limit': query_limit,
            'upgrade_url': 'https://legalmindver1.loca.lt/'
        }), 500

    memory.save_context({'question': question}, {'answer': answer})
    return jsonify({
        'final_response': answer,
        'top_banan_documents': banan_results,
        'chat_history': chat_history_str,
        'related_questions': related_questions,
        'conversation_id': conversation_id,
        'query_count': query_count,
        'query_limit': query_limit
    }), 200

# Soạn thảo bản án
@app.route("/draft_judgment", methods=["POST"])
def draft_judgment():
    if 'user_id' not in session:
        return jsonify({'error': 'Vui lòng đăng nhập'}), 401
    data = request.get_json(silent=True) or {}
    case_details = data.get('case_details', '').strip()
    if not case_details:
        return jsonify({'error': 'Chi tiết vụ án không hợp lệ'}), 400
    banan_results = retrieve(case_details, index, embeddings_data, k=2)
    top_banan_docs = [{'source': r['file'], **r} for r in banan_results]
    chat_history_str = format_chat_history(memory)
    judgment = "Placeholder judgment: Drafted legal document based on case details."
    memory.save_context({'case_details': case_details}, {'judgment': judgment})
    return jsonify({
        'judgment': judgment,
        'top_banan_documents': top_banan_docs,
        'chat_history': chat_history_str
    })

# Xóa hội thoại
@app.route('/conversation/<conversation_id>', methods=['DELETE'])
def delete_conversation(conversation_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Vui lòng đăng nhập'}), 401
    user_id = session['user_id']
    result = db.conversations.delete_one({'_id': ObjectId(conversation_id), 'user_id': user_id})
    if result.deleted_count == 0:
        return jsonify({'error': 'Hội thoại không tồn tại'}), 404
    db.messages.delete_many({'conversation_id': conversation_id})
    return jsonify({'message': 'Xóa hội thoại thành công'}), 200

# Admin routes
@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    users = db.users.find()
    user_name = session.get('username')
    users_list = [{
        'id': str(user['_id']),
        'username': user['username'],
        'email': user['email'],
        'phone': user['phone'],
        'is_active': user.get('is_active', False),
        'is_admin': user.get('is_admin', False),
        'account_type': user.get('account_type', 'limited'),
        'query_limit': user.get('query_limit', None),
        'query_count': user.get('query_count', 0),
        'last_reset': user.get('last_reset', None).isoformat() if user.get('last_reset') else None
    } for user in users]
    return render_template('admin_dashboard.html', users=users_list, user_name = user_name)

@app.route('/admin/users', methods=['GET'])
@admin_required
def get_all_users():
    users = db.users.find()
    return jsonify([{
        'id': str(user['_id']),
        'username': user['username'],
        'email': user['email'],
        'phone': user['phone'],
        'is_active': user.get('is_active', False),
        'is_admin': user.get('is_admin', False),
        'account_type': user.get('account_type', 'limited'),
        'query_limit': user.get('query_limit', None),
        'query_count': user.get('query_count', 0),
        'last_reset': user.get('last_reset', None).isoformat() if user.get('last_reset') else None
    } for user in users]), 200

@app.route('/admin/user/<user_id>', methods=['PUT'])
@admin_required
def update_user(user_id):
    data = request.get_json(silent=True) or {}
    updates = {}
    if 'account_type' in data and data['account_type'] in ['limited', 'unlimited']:
        updates['account_type'] = data['account_type']
        updates['query_limit'] = 10 if data['account_type'] == 'limited' else None
        updates['query_count'] = 0
        updates['last_reset'] = datetime.utcnow()
    if 'is_admin' in data and isinstance(data['is_admin'], bool):
        updates['is_admin'] = data['is_admin']
    if 'query_limit' in data and isinstance(data['query_limit'], int) and data.get('account_type') == 'limited':
        updates['query_limit'] = data['query_limit']
    if not updates:
        return jsonify({'error': 'Không có thông tin cập nhật hợp lệ'}), 400
    result = db.users.update_one(
        {'_id': ObjectId(user_id)},
        {'$set': updates}
    )
    if result.modified_count == 0:
        return jsonify({'error': 'Không tìm thấy người dùng hoặc không có thay đổi'}), 404
    logging.info(f"Admin updated user {user_id}: {updates}")
    # Broadcast updated query count to the user
    if user_id in connected_clients:
        user = db.users.find_one({'_id': ObjectId(user_id)})
        socketio.emit('query_update', {
            'query_count': user.get('query_count', 0),
            'query_limit': user.get('query_limit', 10)
        }, room=connected_clients[user_id])
        logging.info(f"Broadcasted query update to user {user_id} after admin update")
    return jsonify({'message': 'Cập nhật người dùng thành công'}), 200

@app.route('/admin/user/<user_id>/reset_query', methods=['POST'])
@admin_required
def reset_user_query_count(user_id):
    user = db.users.find_one({'_id': ObjectId(user_id)})
    if not user:
        return jsonify({'error': 'Người dùng không tồn tại'}), 404
    if user.get('account_type') == 'unlimited':
        return jsonify({'error': 'Tài khoản không giới hạn không cần reset!'}), 400
    db.users.update_one(
        {'_id': ObjectId(user_id)},
        {'$set': {'query_count': 0, 'last_reset': datetime.utcnow()}}
    )
    logging.info(f"Admin reset query count for user {user_id}")
    # Broadcast updated query count to the user
    if user_id in connected_clients:
        socketio.emit('query_update', {
            'query_count': 0,
            'query_limit': user.get('query_limit', 10)
        }, room=connected_clients[user_id])
        logging.info(f"Broadcasted query update to user {user_id} after reset")
    return jsonify({'message': 'Reset lượt hỏi đáp thành công'}), 200

# Page routes
@app.route('/')
def page_index():
    return render_template('index.html')

@app.route('/home')
def page_home():
    return render_template('home.html')

@app.route('/login')
def login_page():
    return render_template('login.html')

if __name__ == '__main__':
    socketio.run(app, debug=True)  # Use socketio.run instead of app.run