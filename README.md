---

# SchoolAdminWithMachineLearning

A **Flask-based school administration system** enhanced with **machine learning** to predict student enrollment likelihood. The system manages user registration, authentication, and student records, while integrating predictive analytics (using stacking models and LIME explainability) to provide insights into student enrollment decisions.

---

## 📑 Table of Contents

* [Introduction](#introduction)
* [Features](#features)
* [Tech Stack](#tech-stack)
* [Installation](#installation)
* [Configuration](#configuration)
* [Usage](#usage)
* [Machine Learning Integration](#machine-learning-integration)
* [Project Structure](#project-structure)
* [Examples](#examples)
* [Troubleshooting](#troubleshooting)
* [Contributors](#contributors)
* [License](#license)

---

## 🚀 Introduction

SchoolAdminWithMachineLearning combines a **school administration system** with **AI-driven predictive analytics**.
It allows administrators to register students, store records, and run predictive models to evaluate a student’s likelihood of enrollment.

The system also:

* Uses **stacked models** (RandomForest, Logistic Regression, XGBoost, Meta-Model).
* Provides **explainability** with **LIME**.
* Sends **automated welcome emails** to new users.
* Generates **AI-driven explanations** of predictions via the Ollama `gemma:1b` model.

---

## ✨ Features

* 🔐 **User Authentication** (Register, Login, Logout)
* 🧑‍🎓 **Student Records Management**
* 📊 **Machine Learning Enrollment Predictions**
* 🧾 **Explainability Reports with LIME**
* 📬 **Automated Email Notifications**
* 🔒 **Secure Password Hashing**
* 🌍 **CORS-enabled REST API**

---

## 🛠 Tech Stack

* **Backend:** Flask, Flask-SQLAlchemy, Flask-WTF, Flask-CORS
* **Database:** SQLite (configurable via environment variables)
* **Machine Learning:** Scikit-learn, XGBoost, LIME
* **AI Integration:** Ollama (`gemma:1b`) for natural language explanations
* **Emailing:** smtplib (SMTP with Gmail)
* **Others:** Python-dotenv for config, bcrypt/werkzeug for password hashing

---

## 📦 Installation

1. **Clone the repo**

   ```bash
   git clone https://github.com/marcmrnda/SchoolAdminWithMachineLearning.git
   cd SchoolAdminWithMachineLearning
   ```

2. **Create virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Linux/Mac
   venv\Scripts\activate      # On Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare ML Models**
   Place trained `.pkl` models inside `app/AIMODEL/`:

   * `meta_model.pkl`
   * `rf.pkl`
   * `lr.pkl`
   * `xg.pkl`
   * `training_columns.pkl`
   * `X_train_encoded.pkl`

5. **Set up Ollama (`gemma:1b`)**
   Install [Ollama](https://ollama.ai) and pull the model:

   ```bash
   ollama pull gemma:1b
   ```

   Keep Ollama running in the background.

---

## ⚙️ Configuration

Create a `.env` file in the project root:

```env
SECRET_KEY=your-secret-key
DATABASE_URL=sqlite:///school.db
DATABASE_NAME=school.db
DEBUG=True
EMAIL_USER=your-email@gmail.com
EMAIL_PASSWORD=your-app-password
```

---

## ▶️ Usage

Run the Flask app:

```bash
python app.py
```

Default endpoints:

* `/register` → Register a new student
* `/login` → Login user
* `/logout` → Logout
* `/delete/<id>` → Delete a user by ID
* `/` → Home (to be implemented)
* `/admin` → Admin page (to be implemented)

---

## 🤖 Machine Learning Integration

* **Models Used for Prediction:**

  * RandomForest (`rf.pkl`)
  * Logistic Regression (`lr.pkl`)
  * XGBoost (`xg.pkl`)
  * Meta-Model (`meta_model.pkl`)
* **Explainability:** Predictions are explained using **LIME**.
* **Natural Language Reports:**
  After LIME explanations are generated, the system calls **Ollama’s `gemma:1b` model** to create a short, plain-English explanation of why the prediction was made.

Example prediction flow:

1. Student registers → system extracts features.
2. Features are encoded and passed into stacked models.
3. Final prediction + confidence score produced by meta-model.
4. LIME explains feature contributions.
5. **Ollama (`gemma:1b`) generates a human-readable summary.**

---

## 📂 Project Structure

```
├── app.py               # Entry point
├── app/
│   ├── __init__.py      # App factory, DB setup
│   ├── auth.py          # Authentication & ML predictions
│   ├── view.py          # Frontend routes
│   ├── user.py          # Database models (User, Record)
│   └── AIMODEL/         # Trained ML models (not included)
├── requirements.txt     # Python dependencies
└── .env.example         # Example environment file
```

---

## 📖 Examples

* **Register User** → Predicts enrollment likelihood, stores record, sends password via email.
* **Login User** → Validates credentials, creates session.
* **Explainability** → Stores reasoning text for predictions.

---

## 🐛 Troubleshooting

* ❌ **Model files not found** → Ensure `.pkl` models are inside `app/AIMODEL/`.
* ❌ **Email not sending** → Check Gmail App Password setup and `.env` credentials.
* ❌ **Database not created** → Verify `DATABASE_URL` and run the app once to initialize.
* ❌ **No Ollama explanation generated** → Ensure `ollama pull gemma:1b` has been run and Ollama is running.

---

## 👥 Contributors

* **Angel Malaluan**,**Marc Miranda**,**Ian Medina**,**Katrina Pasadilla**,**Kenneth Averion**,**Ameril Mampao**
* **Also known as **Mikay's Angels****

---

## 📜 License

This project currently has **no license file**.
To allow collaboration and usage, consider adding one (e.g., MIT, Apache 2.0).

---
