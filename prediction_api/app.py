from flask import Flask
from api.routes import api_blueprint

app = Flask(__name__)

# Register the API blueprint
app.register_blueprint(api_blueprint)

# Define a root endpoint
@app.route('/')
def index():
    return {"message": "Welcome to the Machine Learning Prediction API"}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
