from .. import db
from sqlalchemy.sql import func

class User(db.Model):
    __tablename__ = "users"
    
    user_id = db.Column(db.Integer, primary_key=True)
    firstName = db.Column(db.String(100), nullable=False)
    middleName = db.Column(db.String(100), nullable=True)  # extra field not in SQL but added
    lastName = db.Column(db.String(100), nullable=False)
    dateOfBirth = db.Column(db.Date, nullable=False)
    sex = db.Column(db.Enum('male', 'female', 'other'), nullable=False)
    mobileNumber = db.Column(db.String(15), nullable=True)  # extra field not in SQL but added
    emailAddress = db.Column(db.String(150), nullable=False, unique=True)
    password = db.Column(db.String(100), nullable=False)
    student_ID = db.Column(db.String(50), unique=True)
    profilePicture = db.Column(db.String(255))
    account_type = db.Column(db.String(10), default="User")
    createdOn = db.Column(db.DateTime(timezone=True), default=func.now())

    # Relationships
    records = db.relationship("Record", backref="user", cascade="all, delete-orphan")


class Record(db.Model):
    __tablename__ = "records"
    
    record_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.user_id"), nullable=False)
    
    campus = db.Column(db.String(255))
    academic_year = db.Column(db.String(50))
    academic_term = db.Column(db.Enum('1st', '2nd', '3rd'))
    course_1st = db.Column(db.String(255))
    course_2nd = db.Column(db.String(255))
    
    birth_city = db.Column(db.String(255))
    birth_province = db.Column(db.String(255))
    birth_country = db.Column(db.String(255))
    gender = db.Column(db.String(50))
    citizen_of = db.Column(db.String(255))
    
    curr_region = db.Column(db.String(255))
    curr_province = db.Column(db.String(255))
    curr_city = db.Column(db.String(255))
    curr_brgy = db.Column(db.String(255))
    curr_street = db.Column(db.String(255))
    curr_postal = db.Column(db.String(50))
    
    per_country = db.Column(db.String(255))
    per_region = db.Column(db.String(255))
    per_province = db.Column(db.String(255))
    per_city = db.Column(db.String(255))
    per_brgy = db.Column(db.String(255))
    per_street = db.Column(db.String(255))
    per_postal = db.Column(db.String(50))
    
    religion = db.Column(db.String(255))
    civil_status = db.Column(db.String(50))
    student_type = db.Column(db.Enum('full-time', 'working', 'other'))
    last_school_attended = db.Column(db.String(255))
    school_type = db.Column(db.Enum('public', 'private', 'other'))
    likelihood = db.Column(db.Float)  # Changed to Float to store percentage
    confidence = db.Column(db.Float)  # New field for confidence score
    reason = db.Column(db.Text)  # New field for reason/explanation
    localOrForeign = db.Column(db.String(50))  # New field for local or foreign student
    age = db.Column(db.Integer)  # New field for age