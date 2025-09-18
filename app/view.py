from flask import render_template,Blueprint,session,redirect,url_for,flash
from .model.user import User

view = Blueprint("view", __name__)

@view.route('/')
def home():
    pass


@view.route('/admin')
def adminPage():    
    pass

@view.route('/unauthorized')
def forbidden():
    pass
