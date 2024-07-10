 
print('Java', 'Frontend_Angular', 'Dotnet', 'Cyber_security', 'Devops', 'Cloud Engineer', 'Python', 
'Mainframe', 'IOT', 'Data Science', 'PHP Developer', 'RPA Developer', 'QA', 'Network Backup Admin', 
'C#', 'Storage Admin', 'Frontend_React', 'Node js', 'MS D365', 'Network Engineer', 'Docker', 
'Digital Transformation', 'Product Owner', 'Project manager', 'SQL', 'SFDC', 'Sitecore Dev', 
'Sharepoint Developer', 'Servicenow Dev', 'AI', 'ML', 'Flutter', 'C++ Developer')
from fuzzywuzzy import fuzz
import json
import PyPDF2
import docx
import random
import os
import fitz 
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gtts import gTTS
import speech_recognition as sr
import time
from io import BytesIO
import random

class QuizBot1:
    df = pd.read_excel('question&Answer.xlsx')

    @staticmethod
    def preprocess_data(df):
        return df

    @staticmethod
    def extract_text_from_pdf(file_path):
        document = fitz.open(file_path)
        text = ""
        for page_num in range(document.page_count):
            page = document.load_page(page_num)
            text += page.get_text()
        return text

    @staticmethod
    def extract_text_from_docx(file_path):
        document = docx.Document(file_path)
        text = ""
        for para in document.paragraphs:
            text += para.text + "\n"
        return text

    @staticmethod
    def extract_keywords_tfidf(text, top_n=2000):
        # Check if text is empty or too short
        if not text.strip():
            return []

        try:
            vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
            tfidf_matrix = vectorizer.fit_transform([text])
            if tfidf_matrix.shape[1] == 0:
                return []

            feature_names = vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.toarray()[0]
            
            # Get top n keywords based on TF-IDF scores
            top_indices = tfidf_scores.argsort()[-top_n:][::-1]
            keywords = [feature_names[i] for i in top_indices]
            return keywords
        except Exception as e:
            print(f"TF-IDF Error: {e}")
            return []
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
            recognizer.adjust_for_ambient_noise(source)
            try:
                audio = recognizer.listen(source, timeout=15)
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
    def compute_similarity(user_answer, correct_answer):
        # model_name = 'bert-large-nli-mean-tokens'
        # model = SentenceTransformer(model_name)
        # embeddings = model.encode([user_answer, correct_answer], convert_to_tensor=True)
        # similarity_score = util.pytorch_cos_sim(embeddings[0], embeddings[1])
        # similarity_score = float(similarity_score.cpu().numpy()) 
        # similarity_score = round(similarity_score * 10, 1)  
        # return similarity_score
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([correct_answer, user_answer])
        similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
        similarity_score = round(similarity_score * 10, 1)
        return similarity_score
    @staticmethod
    def text_to_speech(text):
        if text:
            tts = gTTS(text=text, slow=False)
            audio_buffer = BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_data = audio_buffer.getvalue()
            return audio_data
        else:
            print("No text provided for speech synthesis.")
            return None
    def clean_and_tokenize(text):
    # Remove special characters and convert to lowercase
        text_cleaned = text.lower()
        # Tokenize into words
        words = text_cleaned.split()
        return words
    def generate_ngrams(words, n):
        ngrams = []
        for i in range(len(words) - n + 1):
            ngrams.append(' '.join(words[i:i+n]))
        # print(ngrams)
        return ngrams
    def match_job_description(job_description, existing_technologies1):
    # Clean and tokenize the job description
        words = QuizBot1.clean_and_tokenize(job_description)
        
        # Apply n-grams (here we use unigrams and bigrams)
        unigrams =QuizBot1.generate_ngrams(words, 1)
        bigrams = QuizBot1.generate_ngrams(words, 2)
        
        # Combine unigrams and bigrams
        ngrams = unigrams + bigrams
        
        # Compare n-grams with existing technologies
        matched_technologies = set()
        for ngram in ngrams:
            for tech in existing_technologies1:
                if tech.lower() in ngram:
                    matched_technologies.add(tech)
        return matched_technologies
    def load_existing_technologies(file_path):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                return data  # Assuming the JSON file directly contains a list of technologies
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print("Error: Unable to load existing technologies. Please check if the file exists and contains valid JSON data.")
            return []
    def combine_lists(list1, list2):
    # Combine the lists
        return list(set(list1 + list2))
        
        return unique_combined_list
    def find_partial_matches(unique_words, result):
        partial_matches = set()
        for word in unique_words:
            for r in result:
                if fuzz.partial_ratio(word.lower(), r.lower()) >= 80:
                    partial_matches.add(word)
        return partial_matches
df = QuizBot1.preprocess_data(QuizBot1.df)
# combined_text = ""
# files= ["JD\JD\.net.docx"]
# technologies_json_path = r'technologies.json'
# user = input("input or file: ")
# if user == 'file':
#   # Expecting multiple paths separated by space
#     for file_path in files:
#         if file_path.endswith('.pdf'):
#             combined_text += QuizBot1.extract_text_from_pdf(file_path) + " "
#         elif file_path.endswith('.docx'):
#             combined_text += QuizBot1.extract_text_from_word(file_path) + " "
#         else:
#             print(f"Unsupported file type: {file_path}")
#             continue
# else:
#     user_input = input("paste the text: ")
#     combined_text += user_input
# keywords = QuizBot1.extract_keywords_tfidf(combined_text)
# print(keywords)
# unique_words=unique_sub_categories = df['Sub - Catogory'].unique() 
# matching_words = []
# for category in unique_words:
#     for word in keywords:
#         if category.lower() == word.lower():
#             matching_words.append(category)
# print("Matching subcategory words:", matching_words)
# existing_technologies =QuizBot1.load_existing_technologies(technologies_json_path)

# # Match job description with existing technologies
# matched_technologies =QuizBot1.match_job_description(combined_text, existing_technologies)
# matching_words2=[]
# for tech in matched_technologies:
#     matching_words2.append(tech)
# print("Matching subcategory words:-",matching_words2)
# result = QuizBot1.combine_lists(matching_words, matching_words2)
# print("Final result:", result)
# partial_matches = QuizBot1.find_partial_matches(unique_words, result)
# print("partial_matches:-",partial_matches)
# def main():
#     #sub_category = input('Choose the sub-category (Java/Frontend_Angular/Dotnet/Cyber security/Devops): ')
#     level = float(input('Enter the level (0 to 30): '))
    
#     # Determine the level type based on the input
#     if 0 <= level <= 3:
#         level_type = 'Basic'
#     elif 3 < level <= 7:
#         level_type = 'Intermediate'
#     elif 7 < level <= 30:
#         level_type = 'Expert'
#     else:
#         level_type = 'Invalid input, please enter a value between 0 and 30'
    
#     # subcategory_counters
#     subcategory_counters = {subcategory: 0 for subcategory in partial_matches}
    
#     while True:
#         #Determine which subcategory to select based on the counters
#         min_counter_subcategory = min(subcategory_counters, key=subcategory_counters.get)
#         sub_category = min_counter_subcategory
        
#         # Select a question from the chosen subcategory
#         question, correct_answer, level_type_display, sr_no, sub_category_display = QuizBot1.select_question(df, sub_category,level_type)
        
#         # Increment the counter for the selected subcategory
#         subcategory_counters[sub_category] += 1
        
#         if question is None:
#             print("No questions available for this subcategory and level type. Exiting.")
#             break
#         tts = gTTS(text=question, lang='en')
#         print("Bot: Asking the question...")
#         tts.save("question.mp3")
#         os.system("start question.mp3")
#         print("Question:", question)
#         print("Correct answer:", correct_answer)
#         print('Level type:', level_type_display)
#         print('Sr. No:', sr_no)
#         print('SubCategory:', sub_category_display)
#         # Using gTTS to make the bot ask the question
#         #user_answer = QuizBot.speech_to_text()
#         # if user_answer == "":
#         #    continue

      
#         user_answer = input('Enter the answer: ')
        
#         # Compute similarity between user's answer and correct answer
#         similarity_score = QuizBot1.compute_similarity(user_answer, correct_answer)
#         if similarity_score <= 4:
#             similarity_score = 0
#         print("Your answer:", user_answer)
#         print("Similarity score:", similarity_score, "/10")
#         choice = input("Do you want to continue? (yes/no): ")
#         if choice.lower() != 'yes':
#             break

# if __name__ == "__main__":
#     main()



   
