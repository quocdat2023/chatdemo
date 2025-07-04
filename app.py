from flask import Flask, request, jsonify, session, render_template
from pymongo import MongoClient
from datetime import datetime
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

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'  # Replace with a secure key
# app.config['MONGO_URI'] = 'mongodb://localhost:27017/legal_assistant'  # Replace with your MongoDB URI
app.config['MONGO_URI'] = 'mongodb+srv://itdatit12:NQJWnj7YwbQML8yE@cluster0.pwv2g0y.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0'
# Initialize MongoDB client
mongo = MongoClient(app.config['MONGO_URI'])
db = mongo.get_database('legal_assistant')

# Password hashing function using hashlib
def hash_password(password: str) -> str:
    salt = os.urandom(32)  # Generate a random salt
    hashed = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)  # 100,000 iterations
    password_hash = (salt + hashed).hex()  # Store salt + hash as hex string
    logging.info(f"Generated password hash: {password_hash}")  # Log the hash
    return password_hash

# Password verification function
def verify_password(stored_password: str, provided_password: str) -> bool:
    if not stored_password or not all(c in '0123456789abcdefABCDEF' for c in stored_password):
        logging.error(f"Invalid stored password format: {stored_password}")
        return False
    try:
        stored_bytes = bytes.fromhex(stored_password)
        salt = stored_bytes[:32]  # Extract salt (first 32 bytes)
        stored_hash = stored_bytes[32:]  # Extract hash
        provided_hash = hashlib.pbkdf2_hmac('sha256', provided_password.encode('utf-8'), salt, 100000)
        return stored_hash == provided_hash
    except ValueError as e:
        logging.error(f"Error in verify_password: {e}, stored_password: {stored_password}")
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

# Đăng ký
@app.route('/register', methods=['POST'])
def register():
    data = request.get_json(silent=True) or {}
    username = data.get('username', '').strip()
    email = data.get('email', '').strip()
    password = data.get('password', '').strip()

    if not username or not email or not password:
        return jsonify({'error': 'Thiếu thông tin bắt buộc'}), 400

    if db.users.find_one({'$or': [{'username': username}, {'email': email}]}):
        return jsonify({'error': 'Tên người dùng hoặc email đã tồn tại'}), 400

    password_hash = hash_password(password)
    user = {
        'username': username,
        'email': email,
        'password_hash': password_hash,
        'created_at': datetime.utcnow()
    }
    result = db.users.insert_one(user)
    return jsonify({'message': 'Đăng ký thành công'}), 201

@app.route('/')
def login_pages():
    return render_template('index.html')

@app.route('/login')
def login_page():
    return render_template('login.html')

@app.route('/register')
def register_page():
    return render_template('register.html')

# Đăng nhập
@app.route('/logins', methods=['POST'])
def login():
    data = request.get_json(silent=True) or {}
    email = data.get('email', '').strip()
    password = data.get('password', '').strip()

    user = db.users.find_one({'email': email})
    if not user:
        logging.error(f"No user found for email: {email}")
        return jsonify({'error': 'Email hoặc mật khẩu không đúng'}), 401

    if not verify_password(user['password_hash'], password):
        return jsonify({'error': 'Email hoặc mật khẩu không đúng'}), 401

    session['user_id'] = str(user['_id'])
    session['username'] = user['username']
    return jsonify({'message': 'Đăng nhập thành công', 'username': user['username']}), 200

# Đăng xuất
@app.route('/logout', methods=['POST'])
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    return jsonify({'message': 'Đăng xuất thành công'}), 200

# Kiểm tra session
@app.route('/check_session', methods=['GET'])
def check_session():
    if 'user_id' in session:
        return jsonify({'logged_in': True, 'username': session['username']}), 200
    return jsonify({'logged_in': False}), 200

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
        return jsonify({'error': 'Vui lòng đăng nhập để sử dụng tính năng này'}), 401

    user_id = session['user_id']
    data = request.get_json(silent=True) or {}
    question = data.get('question', '').strip()
    if not question:
        return jsonify({'error': 'Câu hỏi không hợp lệ'}), 400

    # Tạo hoặc lấy hội thoại hiện tại
    conversation_id = data.get('conversation_id')
    if conversation_id:
        conversation = db.conversations.find_one({'_id': ObjectId(conversation_id), 'user_id': user_id})
        if not conversation:
            return jsonify({'error': 'Hội thoại không tồn tại'}), 404
    else:
        conversation = {
            'user_id': user_id,
            'title': question[:50],
            'timestamp': datetime.utcnow(),
            'messages': []
        }
        result = db.conversations.insert_one(conversation)
        conversation_id = str(result.inserted_id)

    # Lưu tin nhắn của người dùng
    user_message = {
        'conversation_id': conversation_id,
        'type': 'user',
        'content': question,
        'timestamp': datetime.utcnow()
    }
    db.messages.insert_one(user_message)

    # Query dữ liệu tham khảo
    banan_results = retrieve(question, index, embeddings_data, k=5)
    chat_history_str = format_chat_history(memory)

    # Prompt chính
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
   - Nếu Dark mode: Nếu không có bản án, nêu rõ sẽ phân tích trên cơ sở các điều luật hiện hành tại Việt Nam.

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
        return jsonify({"error": "Lỗi khi gọi Gemini: " + str(e)}), 500

    # Prompt câu hỏi liên quan
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
        return jsonify({"error": "Lỗi khi gọi Gemini: " + str(e)}), 500

    related_questions = preprocess_related_questions(related_questions)

    # Lưu câu trả lời của trợ lý
    assistant_message = {
        'conversation_id': conversation_id,
        'type': 'assistant',
        'content': answer,
        'timestamp': datetime.utcnow(),
        'sources': banan_results,
        'related_questions': related_questions
    }
    db.messages.insert_one(assistant_message)

    # Cập nhật tiêu đề hội thoại
    db.conversations.update_one(
        {'_id': ObjectId(conversation_id)},
        {'$set': {'title': question[:50], 'timestamp': datetime.utcnow()}}
    )

    # Lưu vào memory
    memory.save_context({'question': question}, {'answer': answer})

    return jsonify({
        'final_response': answer,
        'top_banan_documents': banan_results,
        'chat_history': chat_history_str,
        'related_questions': related_questions,
        'conversation_id': conversation_id
    })

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

if __name__ == '__main__':
    app.run(debug=True)