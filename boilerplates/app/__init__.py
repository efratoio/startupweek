import os
from flask import Flask
from flask_cors import CORS

app = Flask(__name__)

# enable CORS
CORS(app)
# set config
app_settings = os.getenv('APP_SETTINGS')
app.config.from_object(app_settings)
print(app.config)

from app import views