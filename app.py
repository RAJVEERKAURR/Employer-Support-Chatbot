from flask import Flask, request, jsonify, render_template
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords

# Downloading stopwords
nltk.download('stopwords')
stop_words = list(stopwords.words('english'))  # Convert set to list

app = Flask(__name__)

# Load your dataset
df = pd.read_excel("Course_Dataset.xlsx")

# Function to clean and preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

# Chatbot implementation
class Chatbot:
    def __init__(self, dataframe):
        self.df = dataframe
        self.step = 0
        self.job_title = ''
        self.job_skills = ''
        self.job_duties = ''
        
    def process_input(self, user_input):
        responses = []
        
        if user_input.lower() == 'quit':
            responses.append('Thank you for using Humber Bot. Goodbye!')
            self.step = 0
            return responses
        
        if self.step == 0:
            if user_input.lower() == "yes":
                self.step += 1
                responses.append('Please enter the job title.')
            else:
                responses.append('I currently cannot provide any output for employers. I\'m only designed for Humber staff.')
        elif self.step == 1:
            self.job_title = user_input
            responses.append('Please type all the job skills.')
            self.step += 1
        elif self.step == 2:
            self.job_skills = user_input
            responses.append('Please type the main job duties.')
            self.step += 1
        elif self.step == 3:
            self.job_duties = user_input
            responses.append('Please wait while we look up the appropriate programs.')
            program_responses = self.find_programs()
            responses.extend(program_responses)
            responses.append('Do you want more recommendations or quit? (Type "more" or "quit")')
            self.step = 4
        elif self.step == 4:
            if user_input.lower() == 'more':
                self.step = 0
                responses.append('Are you Humber staff?')
            elif user_input.lower() == 'quit':
                responses.append('Thank you for using Humber Bot. Goodbye!')
                self.step = 0
            else:
                responses.append('Invalid input. Please type "more" or "quit".')
        
        return responses

    def find_programs(self):
        combined_job_desc = f"{self.job_title} {self.job_skills} {self.job_duties}"
        combined_job_desc = preprocess_text(combined_job_desc)
        
        tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words)
        tfidf_matrix = tfidf_vectorizer.fit_transform(self.df['PROGRAM OVERVIEW'].apply(preprocess_text))
        
        job_desc_vector = tfidf_vectorizer.transform([combined_job_desc])
        cosine_similarities = cosine_similarity(job_desc_vector, tfidf_matrix).flatten()
        
        self.df['Match Score'] = cosine_similarities
        top_programs = self.df.nlargest(5, 'Match Score')
        
        responses = []
        for _, row in top_programs.iterrows():
            keywords = self.extract_keywords(combined_job_desc, row['PROGRAM OVERVIEW'], tfidf_vectorizer)
            
            # Check Work Integrated Learning column
            work_integrated_learning = "No"
            if row['WORK INTEGRATED LEARNING'].strip().lower() != 'no':
                work_integrated_learning = f"Yes\nCourse Code: {row['COURSE CODE']}"
            
            response = (
                f"Program Name: {row['PROGRAM NAME']}\n"
                f"Match Score: {row['Match Score']:.2f}\n"
                f"Work Integrated Learning: {work_integrated_learning}\n"
                f"Program Code: {row['CODE']}\n"
                f"Credentials: {row['CREDENTIALS']}\n"
                f"Faculty: {row['FACULTY']}\n"
                f"Matched Keywords: {', '.join(keywords)}"
            )
            responses.append(response)
        
        return responses

    def extract_keywords(self, job_desc, program_overview, vectorizer):
        job_desc_terms = set(preprocess_text(job_desc).split()) - set(stop_words)
        program_terms = set(preprocess_text(program_overview).split()) - set(stop_words)
        common_terms = job_desc_terms.intersection(program_terms)

        tfidf_feature_names = vectorizer.get_feature_names_out()
        job_desc_tfidf = vectorizer.transform([job_desc]).toarray().flatten()
        program_tfidf = vectorizer.transform([program_overview]).toarray().flatten()
        
        common_keywords = []
        for term in common_terms:
            if term in tfidf_feature_names:
                term_index = tfidf_feature_names.tolist().index(term)
                if job_desc_tfidf[term_index] > 0 and program_tfidf[term_index] > 0:
                    common_keywords.append((term, job_desc_tfidf[term_index] + program_tfidf[term_index]))
        
        # Sort keywords by their combined TF-IDF score and return the top 10
        common_keywords = sorted(common_keywords, key=lambda x: x[1], reverse=True)[:10]
        return [term for term, _ in common_keywords]

chatbot = Chatbot(df)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get("message")
    responses = chatbot.process_input(user_input)
    return jsonify({"responses": responses})

if __name__ == '__main__':
    app.run(debug=True)
