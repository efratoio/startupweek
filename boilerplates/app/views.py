from app import app, api
from flask import jsonify


BTC = "BTC"

CHINA = "CNY"


@app.route('/graph', methods=['GET'])
def digit_month():
    return jsonify(
        api.get_digit_curr_monthly(CHINA,BTC)
        )