# 🧠 Smart Admission: Enrollment & Reenrollment Prediction System

A **Flask-based school administration system** enhanced with **machine learning** to predict a student's **enrollment or reenrollment likelihood**. The system manages user registration, authentication, and student records, while integrating predictive analytics (using stacking models and LIME explainability) to provide actionable insights into student commitment decisions.

-----

## 📑 Table of Contents

  * [Introduction](https://www.google.com/search?q=%23introduction)
  * [Features](https://www.google.com/search?q=%23features)
  * [Tech Stack](https://www.google.com/search?q=%23tech-stack)
  * [Installation](https://www.google.com/search?q=%23installation)
  * [Configuration](https://www.google.com/search?q=%23configuration)
  * [Usage](https://www.google.com/search?q=%23usage)
  * [Machine Learning Integration](https://www.google.com/search?q=%23machine-learning-integration)
  * [Project Structure](https://www.google.com/search?q=%23project-structure)
  * [Examples](https://www.google.com/search?q=%23examples)
  * [Troubleshooting](https://www.google.com/search?q=%23troubleshooting)
  * [Contributors](https://www.google.com/search?q=%23contributors)
  * [License](https://www.google.com/search?q=%23license)

-----

## 🚀 Introduction

**Smart Admission** combines a **school administration system** with **AI-driven predictive analytics** focused on student retention and commitment.

It allows administrators to register students, store records, and run predictive models to evaluate a student’s **likelihood of initial enrollment or reenrollment** for the next term.

The system also:

  * Uses **stacked models** (RandomForest, Logistic Regression, XGBoost, Meta-Model) for higher predictive accuracy.
  * Provides **explainability** with **LIME** to understand *why* a prediction was made.
  * Sends **automated welcome emails** to new users.
  * Generates **AI-driven explanations** of predictions via the Ollama `gemma:1b` model, translating complex data into plain English.

-----

## ✨ Features

  * 🔐 **User Authentication** (Register, Login, Logout)
  * 🧑‍🎓 **Student Records Management**
  * 📊 **Machine Learning Enrollment/Reenrollment Predictions**
  * 🧾 **Explainability Reports with LIME**
  * 📬 **Automated Email Notifications**
  * 🔒 **Secure Password Hashing**
  * 🌍 **CORS-enabled REST API**

-----

## 🛠 Tech Stack

  * **Backend:** Flask, Flask-SQLAlchemy, Flask-WTF, Flask-CORS
  * **Database:** SQLite (configurable via environment variables)
  * **Machine Learning:** Scikit-learn, XGBoost, LIME
  * **AI Integration:** Ollama (`gemma:1b`) for natural language explanations
  * **Emailing:** smtplib (SMTP with Gmail)
  * **Others:** Python-dotenv for config, bcrypt/werkzeug for password hashing

-----

## 📦 Installation

1.  **Clone the repo**

    ```bash
    git clone https://github.com/marcmrnda/SchoolAdminWithMachineLearning.git
    cd SchoolAdminWithMachineLearning
    ```

2.  **Create and activate Conda environment (Required)**

    You must use **Conda** to manage the environment for the machine learning dependencies.

    ```bash
    # Create the environment with a modern Python version
    conda create -n smart-admission python=3.9 
    # Activate the environment
    conda activate smart-admission
    ```

    *OR, use venv (Optional alternative to Conda)*

    ```bash
    python -m venv venv
    source venv/bin/activate    # On Linux/Mac
    venv\Scripts\activate       # On Windows
    ```

3.  **Install dependencies**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Prepare ML Models**
    Place trained `.pkl` models inside `app/AIMODEL/`:

      * `meta_model.pkl`
      * `rf.pkl`
      * `lr.pkl`
      * `xg.pkl`
      * `training_columns.pkl`
      * `X_train_encoded.pkl`

5.  **Set up Ollama (`gemma:1b`)**
    Install [Ollama](https://ollama.ai) and pull the model:

    ```bash
    ollama pull gemma:1b
    ```

    Keep Ollama running in the background.

-----

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

-----

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

-----

## 🤖 Machine Learning Integration

The core of Smart Admission is its predictive capability, focused on student commitment.

  * **Target Prediction:** The system predicts the likelihood of a student to **enroll** (for applicants) or **reenroll** (for existing students) based on their features.

  * **Models Used for Prediction:**

      * RandomForest (`rf.pkl`)
      * Logistic Regression (`lr.pkl`)
      * XGBoost (`xg.pkl`)
      * **Meta-Model** (`meta_model.pkl`) - Combines the outputs of the base models for a final, highly reliable prediction.

  * **Explainability:** Predictions are explained using **LIME**, highlighting which student factors (e.g., previous grades, distance from school) contributed most to the prediction.

  * **Natural Language Reports:**
    The system utilizes **Ollama’s `gemma:1b` model** to take the raw LIME explanation and generate a short, plain-English summary for administrators.

Example prediction flow:

1.  Student registers/record is updated → system extracts features.
2.  Features are encoded and passed into stacked models.
3.  Final prediction + confidence score produced by meta-model.
4.  LIME explains feature contributions.
5.  **Ollama (`gemma:1b`) generates a human-readable summary.**

-----

## 📂 Project Structure

```
├── app.py              # Entry point
├── app/
│   ├── __init__.py     # App factory, DB setup
│   ├── auth.py         # Authentication & ML predictions
│   ├── view.py         # Frontend routes
│   ├── user.py         # Database models (User, Record)
│   └── AIMODEL/        # Trained ML models (not included)
├── requirements.txt    # Python dependencies
└── .env.example        # Example environment file
```

-----

## 📖 Examples

  * **Register User** → Predicts enrollment likelihood, stores record, sends password via email.
  * **Login User** → Validates credentials, creates session.
  * **Explainability** → Stores reasoning text for predictions.

-----

## 🐛 Troubleshooting

  * ❌ **Model files not found** → Ensure `.pkl` models are inside `app/AIMODEL/`.
  * ❌ **Email not sending** → Check Gmail App Password setup and `.env` credentials.
  * ❌ **Database not created** → Verify `DATABASE_URL` and run the app once to initialize.
  * ❌ **No Ollama explanation generated** → Ensure `ollama pull gemma:1b` has been run and Ollama is running.

-----

## 👥 Contributors

  * **Angel Malaluan**,**Marc Miranda**,**Ian Medina**,**Katrina Pasadilla**,**Kenneth Averion**,**Ameril Mampao**
  * **Also known as Mikay's Angels**

-----

## 📜 License

This project currently has **no license file**.
To allow collaboration and usage, consider adding one (e.g., MIT, Apache 2.0).

-----
