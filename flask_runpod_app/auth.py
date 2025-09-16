"""
User Authentication System
Adapted from testw4 with hardcoded users
"""
from flask_login import UserMixin
from models import db, User

# Hardcoded users for internal use (10 users)
HARDCODED_USERS = [
    {
        'username': 'admin',
        'password': 'admin123',
        'email': 'vaclavik.renturi@gmail.com',
        'is_admin': True
    },
    {
        'username': 'user1',
        'password': 'user123',
        'email': 'vaclavik.renturi@gmail.com',
        'is_admin': False
    },
    {
        'username': 'user2',
        'password': 'user123',
        'email': 'vaclavik.renturi@gmail.com',
        'is_admin': False
    },
    {
        'username': 'user3',
        'password': 'user123',
        'email': 'vaclavik.renturi@gmail.com',
        'is_admin': False
    },
    {
        'username': 'user4',
        'password': 'user123',
        'email': 'vaclavik.renturi@gmail.com',
        'is_admin': False
    },
    {
        'username': 'user5',
        'password': 'user123',
        'email': 'vaclavik.renturi@gmail.com',
        'is_admin': False
    },
    {
        'username': 'user6',
        'password': 'user123',
        'email': 'vaclavik.renturi@gmail.com',
        'is_admin': False
    },
    {
        'username': 'user7',
        'password': 'user123',
        'email': 'vaclavik.renturi@gmail.com',
        'is_admin': False
    },
    {
        'username': 'user8',
        'password': 'user123',
        'email': 'vaclavik.renturi@gmail.com',
        'is_admin': False
    },
    {
        'username': 'demo',
        'password': 'demo123',
        'email': 'vaclavik.renturi@gmail.com',
        'is_admin': False
    }
]


class AuthUser(UserMixin):
    """User class for Flask-Login"""
    def __init__(self, user_db):
        self.id = user_db.id
        self.username = user_db.username
        self.email = user_db.email
        self.is_admin = user_db.is_admin
        self._user_db = user_db
    
    def get_id(self):
        return str(self.id)
    
    @property
    def is_authenticated(self):
        return True
    
    @property
    def is_active(self):
        return True
    
    @property
    def is_anonymous(self):
        return False


def init_auth(app):
    """Initialize authentication with hardcoded users"""
    with app.app_context():
        # Create users if they don't exist
        for user_data in HARDCODED_USERS:
            user = User.query.filter_by(username=user_data['username']).first()
            if not user:
                user = User(
                    username=user_data['username'],
                    email=user_data['email'],
                    is_admin=user_data.get('is_admin', False)
                )
                user.set_password(user_data['password'])
                db.session.add(user)
                print(f"Created user: {user_data['username']}")
        
        db.session.commit()
        print("Authentication initialized with hardcoded users")


def authenticate_user(username, password):
    """Authenticate user with username and password"""
    user = User.query.filter_by(username=username).first()
    if user and user.check_password(password):
        return AuthUser(user)
    return None


def load_user(user_id):
    """Load user by ID for Flask-Login"""
    try:
        user = User.query.get(int(user_id))
        if user:
            return AuthUser(user)
    except (ValueError, TypeError):
        pass
    return None