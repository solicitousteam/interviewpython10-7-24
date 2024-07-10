from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gtts import gTTS
import os
import speech_recognition as sr
import time
from io import BytesIO
from werkzeug.utils import secure_filename
import base64
from JD import QuizBot1
import tempfile
import fitz
import random
import docx
import numpy as np 
app = Flask(__name__)
CORS(app)  # Enable CORS
app.config['UPLOAD_FOLDER'] = '/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'docx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

class QuizBot:
    df = pd.read_excel('question&Answer.xlsx')

    @staticmethod
    def preprocess_data(df):
        return df
    @staticmethod
    def select_question(df, sub_category, level_type):
        filtered_df = df[(df['Sub - Catogory'] == sub_category) & (df['Level Type'] == level_type)]
        if filtered_df.empty:
            return None, None, None, None, None
        else:
            question_row = filtered_df.sample(n=1)
            question = question_row['Question'].values[0]
            correct_answer = question_row['Answer'].values[0]
            level_type_display = question_row['Level Type'].values[0]
            sr_no = question_row['Sr. No'].values[0]
            sub_category_display = question_row['Sub - Catogory'].values[0]
            return question, correct_answer, level_type_display, sr_no, sub_category_display
    @staticmethod
    def speech_to_text():
        recognizer = sr.Recognizer()
        print("Waiting for 15 seconds before starting speech recognition...")
        time.sleep(15)
        
        with sr.Microphone() as source:
            print("Listening...")
            recognizer.adjust_for_ambient_noise(source)  # Adjust for ambient noise
            try:
                audio = recognizer.listen(source, timeout=10)  # Stop listening after 10 seconds of speech
            except sr.WaitTimeoutError:
                print("No speech detected after 10 seconds. Stopping...")
                return ""
        
        try:
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            print("Sorry, I could not understand the audio.")
            return ""
        except sr.RequestError as e:
            print("Sorry, there was an error processing your request:", str(e))
            return ""
    @staticmethod
    def compute_similarity(user_answer, dataset_answer):
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([dataset_answer, user_answer])
        similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
        return similarity_score

    @staticmethod
    def text_to_speech(text):
        if text:
            tts = gTTS(text=text,slow=False)
            audio_buffer = BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_data = audio_buffer.getvalue()
            return audio_data
        else:
            # Handle the case where no text is provided
            print("No text provided for speech synthesis.")
            return None

df = QuizBot.preprocess_data(QuizBot.df)

technologies_json_path=r'technologies.json'
def extract_text_from_pdf(file_path):
    document = fitz.open(file_path)
    text = ""
    for page_num in range(document.page_count):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

def extract_text_from_docx(file_path):
    document = docx.Document(file_path)
    text = ""
    for para in document.paragraphs:
        text += para.text + "\n"
    return text
def extract_keywords_tfidf(text, top_n=2000):
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray()[0]
        
        top_indices = tfidf_scores.argsort()[-top_n:][::-1]
        keywords = [feature_names[i] for i in top_indices]
        return keywords
@app.route('/question', methods=['POST'])
def get_question():
    if request.method=='POST':
        data = request.json
        sub_category = data['sub_category']
        level_type = data['level_type']
        question, correct_answer, level_type_display, sr_no, sub_category_display = QuizBot.select_question(df, sub_category, level_type)
        
        if question is None:
            return jsonify({"error": "No questions found for the selected sub-category and level type."}), 404
        
        audio=QuizBot.text_to_speech(question)
        if audio:
            audio_base64 = base64.b64encode(audio).decode('utf-8')
        else:
            audio_base64 = ''
        return jsonify({
            "question": question,
            "correct_answer": correct_answer,
            "audio_base64":audio_base64
            
        })
    elif request.method=='GET':
        return jsonify({"massege": 'successfull'})
        

@app.route('/answer', methods=['POST'])
def check_answer():
    data = request.json
    user_answer = data['user_answer']
    correct_answer = data['correct_answer']
    similarity_score = QuizBot.compute_similarity(user_answer, correct_answer)
    similarity_score = round(similarity_score * 10, 1)

    if similarity_score < 3:
        similarity_score = 0
    
    return jsonify({
        "user_answer": user_answer,
        "correct_answer": correct_answer,
        "similarity_score": similarity_score
    })
@app.route('/custom_interview', methods=['POST','GET'])
def upload_JD():
    global level_type, partial_matches, df
    text=''
    if request.method=="POST":
        if 'file' not in request.files and 'text' not in request.form:
            return jsonify({'error': 'No file or text provided'}), 400

        file = request.files.get('file')
        pasted_text = request.form.get('text')
        level = request.form.get('level_of_experience')

        if not level:
            return jsonify({'error': 'Level of experience not provided'}), 400

        try:
            level = float(level)
        except ValueError:
            return jsonify({'error': 'Invalid level_of_experience'}), 400

        text = pasted_text

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])
            file.save(file_path)

            if filename.endswith('.pdf'):
                text = extract_text_from_pdf(file_path)
            elif filename.endswith('.docx'):
                text = extract_text_from_docx(file_path)
            else:
                return jsonify({'error': 'Unsupported file type'}), 400

            os.remove(file_path)

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        # Rest of your logic...

        keywords = extract_keywords_tfidf(text, top_n=2000)
        unique_words = df['Sub - Catogory'].unique()
        unique_words = unique_words.tolist()

        matching_words = []
        for category in unique_words:
            for word in keywords:
                if category.lower() == word.lower():
                    matching_words.append(category)

        existing_technologies = QuizBot1.load_existing_technologies(technologies_json_path)
        matched_technologies = QuizBot1.match_job_description(text, existing_technologies)
        matching_words2 = []
        for tech in matched_technologies:
            matching_words2.append(tech)

        result = QuizBot1.combine_lists(matching_words, matching_words2)
        partial_matches = QuizBot1.find_partial_matches(unique_words, result)

        if 0 <= level <= 3:
            level_type = 'Basic'
        elif 3 < level <= 7:
            level_type = 'Intermediate'
        elif 7 < level <= 30:
            level_type = 'Expert'
        else:
            return jsonify({'error': 'Invalid input. Please enter a value between 0 and 30'}), 400

        # subcategory_counters = {subcategory: 0 for subcategory in partial_matches}

        # while True:
        #     sub_category = random.choice(list(partial_matches))

        #     question, correct_answer, level_type_display, sr_no, sub_category_display = QuizBot1.select_question(df, sub_category, level_type=level_type)
        #     if question is not None:
        #         break  # Exit the loop if a question is found

        # if question is None:
        #     return jsonify({"error": "No questions found for the selected sub-category and level type."}), 404

        # audio = QuizBot1.text_to_speech(question)
        # if audio:
        #     audio_base64 = base64.b64encode(audio).decode('utf-8')
        # else:
        #     audio_base64 = ''

        # # Convert any sets to lists before returning the JSON response
        # partial_matches_list = list(partial_matches) if isinstance(partial_matches, set) else partial_matches

        # return jsonify({
        #     'text': text,
        #     'partial_matches': partial_matches_list,
        #     'question': question,
        #     'correct_answer': correct_answer,
        #     'audio': audio_base64
        # }), 200
        return jsonify({
        'message': 'Job description processed successfully. You can now request questions.',
        'text': text,
        'level_type': level_type,
        'partial_matches': list(partial_matches)
    }), 200
 
@app.route('/next_question', methods=['GET'])
def next_question():
    global level_type, partial_matches, df
    
    if level_type is None or not partial_matches or df.empty:
        return jsonify({'error': 'The job description has not been processed yet.'}), 400
    
    sub_category=random.choice(list(partial_matches))

    question, correct_answer, level_type_display, sr_no, sub_category_display = QuizBot1.select_question(df, sub_category, level_type=level_type)
    
    if question is None:
        return jsonify({"error": "No questions found for the selected sub-category and level type."}), 404

    audio = QuizBot1.text_to_speech(question)
    if audio:
        audio_base64 = base64.b64encode(audio).decode('utf-8')
    else:
        audio_base64 = ''

    return jsonify({
        'question': question,
        'correct_answer': correct_answer,
        'audio': audio_base64
    }), 200

@app.route('/custom_answer', methods=['POST'])
def custom_check_answer():
    data = request.json
    user_answer = data['user_answer']
    correct_answer = data['correct_answer']
    similarity_score = QuizBot.compute_similarity(user_answer, correct_answer)
    similarity_score = round(similarity_score * 10, 1)

    if similarity_score < 3:
        similarity_score = 0
    
    return jsonify({
        "user_answer": user_answer,
        "correct_answer": correct_answer,
        "similarity_score": similarity_score
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5000)
