from flask import Flask, render_template, send_from_directory
import os

app = Flask(__name__, template_folder='templates')

@app.route('/')
def index():
    return render_template('bailanindex.html')

@app.route('/static/<path:path>')
def static_files(path):
    return send_from_directory('.', path)

if __name__ == '__main__':
    app.run(debug=True, port=5003, host='0.0.0.0')
