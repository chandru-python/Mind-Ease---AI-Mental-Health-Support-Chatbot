# 🧠 Mind Ease - AI Mental Health Support Chatbot

Mind Ease is an **AI-powered mental health support chatbot** built using **Flask, Deep Learning, and NLP**.  
It detects emotions from user text input and responds with supportive, empathetic replies along with coping tips, guided suggestions, mood tracking, and dashboard analytics.

This project is designed to provide a safe, interactive, and intelligent emotional support experience through a web-based chatbot system.

---

## 🚀 Features

- 💬 Real-time chatbot conversation
- 🧠 Emotion detection using Deep Learning
- 📊 Mood tracking dashboard
- 📝 Conversation history logging
- 🚨 Crisis input detection
- 🎯 Confidence score for detected emotions
- 🌿 Personalized coping tips
- 🎥 Guided video suggestions
- 🔐 User login and registration system
- 📈 Performance metrics visualization

---

## 🛠️ Tech Stack

### Backend
- Python
- Flask
- SQLite

### Machine Learning / NLP
- TensorFlow / Keras
- Scikit-learn
- NumPy
- Pandas

### Frontend
- HTML
- CSS
- JavaScript

---

## 📂 Project Structure

```bash
Mind-Ease---AI-Mental-Health-Support-Chatbot/
│
├── app.py
├── training.py
├── testing.py
├── data.json
├── users.db
├── README.md
├── requirements.txt
│
├── templates/
│   ├── index.html
│   ├── about.html
│   ├── login.html
│   ├── register.html
│   ├── chatbot.html
│   ├── dashboard.html
│   └── metrics.html
│
├── static/
│   ├── accuracy.png
│   ├── loss.png
│   ├── metrics.png
│   └── roc.png
│
└── model files (not uploaded)
    ├── emotion_model.keras
    ├── tokenizer.pkl
    ├── label_encoder.pkl
    └── glove.6B.100d.txt
```

---

## 🧠 How It Works

The chatbot takes **user text input** and processes it through a trained **emotion classification model**.  
The system then:

1. Cleans and preprocesses the text
2. Detects the user’s emotional state
3. Matches the input with suitable supportive responses
4. Provides coping suggestions and video guidance
5. Logs the mood and conversation into the database
6. Displays mood insights on the dashboard

The chatbot can recognize emotional categories such as:

- Sad
- Anxiety
- Anger
- Happy
- Neutral
- Crisis / Distress situations

---

## 📊 Dashboard Features

The dashboard helps visualize emotional patterns over time.

### Included Dashboard Components:
- Emotion frequency chart
- Confidence-based mood tracking
- Recent mood log table
- Recent chatbot conversation history
- Model metrics page

This makes the project more interactive and useful for understanding emotional trends.

---

## 🚨 Crisis Detection

Mind Ease includes a simple **crisis detection mechanism**.  
If the user enters text indicating possible emotional crisis or distress, the chatbot responds with a more serious and supportive message instead of a normal emotional reply.

This improves the safety and responsibility of the chatbot system.

---

## 🔐 Authentication System

The project includes a complete user authentication module:

- User Registration
- User Login
- Secure Password Hashing
- User Session Management
- Logout Functionality

Each user can maintain their own conversation and mood history securely.

---

## 📌 Model Files

Some trained model files are **not uploaded to GitHub** because of file size limitations.

### Required files to run the project:
- `emotion_model.keras`
- `tokenizer.pkl`
- `label_encoder.pkl`
- `glove.6B.100d.txt`

Please place these files in the **root project folder** before running the application.

---

## ▶️ How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/chandru-python/Mind-Ease---AI-Mental-Health-Support-Chatbot.git
cd Mind-Ease---AI-Mental-Health-Support-Chatbot
```

---

### 2️⃣ Create Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
venv\Scripts\activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Add Required Model Files

Make sure these files are present in your project root:

```bash
emotion_model.keras
tokenizer.pkl
label_encoder.pkl
glove.6B.100d.txt
```

---

### 5️⃣ Run the Application

```bash
python app.py
```

---

### 6️⃣ Open in Browser

```bash
http://127.0.0.1:5000
```

---

## 📷 Output Screens

This repository also includes generated result plots such as:

- Accuracy graph
- Loss graph
- Metrics graph
- ROC curve

These are stored inside the `static/` folder.

---

## 📈 Future Improvements

Possible future enhancements for this project:

- Voice-based emotion detection
- Multilingual support
- More advanced transformer-based NLP models
- Better crisis escalation system
- Mobile-friendly UI improvements
- Admin panel for monitoring
- Integration with therapist or helpline suggestions

---

## 🎯 Learning Outcomes

This project helped strengthen practical skills in:

- Flask web development
- NLP-based emotion understanding
- Deep learning model deployment
- SQLite database handling
- Frontend-backend integration
- AI application design for real-world use

---

## 👨‍💻 Author

**Chandru**  
Machine Learning | Deep Learning | AI Projects | Web-based Intelligent Systems

GitHub: [chandru-python](https://github.com/chandru-python)

---

## 📌 Repository Description

**An AI-powered mental health support chatbot built with Flask, Deep Learning, and NLP. It detects user emotions from text, provides supportive responses, coping tips, mood tracking, crisis detection, and dashboard analytics.**

---

## ⭐ If you like this project

If you found this project useful, feel free to:

- ⭐ Star this repository
- 🍴 Fork it
- 🛠️ Improve it
- 📢 Share it

---

## ⚠️ Disclaimer

This project is created for **educational and research purposes only**.  
It is **not a replacement for professional mental health care, therapy, or emergency support**.

If someone is in immediate danger or crisis, they should contact a qualified mental health professional or emergency service.
