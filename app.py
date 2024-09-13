from flask import Flask, render_template, request, redirect, url_for, session
from setup import make_predictions
from werkzeug.utils import secure_filename
from pathlib import Path

app = Flask(__name__, static_folder='staticfiles')
upload_folder = Path('staticfiles/uploads')
upload_folder.mkdir(exist_ok=True, parents=True)
app.secret_key = 'my_secret_key'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('image')
        filename = secure_filename(file.filename)
        file_save_path = Path(upload_folder / filename)
        # if not file_save_path.exists():
        #     file.save(file_save_path)
        # session['image_path'] = str(file_save_path)
        prediction = make_predictions(image=file)
        context = {
            'file_path': file_save_path,
            'prediction': prediction
        }
        return render_template('home.html', context=context)
    return render_template('home.html')


if __name__ == '__main__':
    app.run(debug=True, )