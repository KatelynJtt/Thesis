from flask import Blueprint, render_template, request, jsonify

# Create a Blueprint for the preprocessing app
preprocessing_bp = Blueprint('preprocessing', __name__, template_folder='blueprints/templates/preprocessing', static_folder='blueprints/static/css')

# Define a route for the preprocessing page
@preprocessing_bp.route('/')
def preprocessing():
    return render_template('preprocessing.html')

