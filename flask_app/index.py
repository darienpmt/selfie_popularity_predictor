import os
import sys
import random
from flask import Flask, flash, render_template, redirect, request, url_for, send_file
from werkzeug.utils import secure_filename
# viz imports
from numpy import pi
from bokeh.plotting import figure
from bokeh.embed import components

# custom function imports
from main import generate_barplot, generate_random_name, is_allowed_file, load_and_prepare, make_thumbnail

sys.path.insert(0, os.path.realpath(os.path.dirname(__file__)))
os.chdir(os.path.realpath(os.path.dirname(__file__)))

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ['SECRET_KEY']
app.config['UPLOAD_FOLDER'] = os.environ['UPLOAD_FOLDER']

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        # show the upload form
        return render_template('home.html')

    if request.method == 'POST':
        # check if a file was passed into the POST request
        if 'image' not in request.files:
            flash('No file was uploaded.')
            return redirect(request.url)

        image_file = request.files['image']

        # if filename is empty, then assume no upload
        if image_file.filename == '':
            flash('No file was uploaded.')
            return redirect(request.url)

        # if the file is "legit"
        if image_file and is_allowed_file(image_file.filename):
            passed = False
            try:
                filename = generate_random_name(image_file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                image_file.save(filepath)
                passed = make_thumbnail(filepath)
            except Exception:
                passed = False

            if passed:
                return redirect(url_for('predict', filename=filename))
            else:
                flash('An error occurred, try again.')
                return redirect(request.url)

@app.route('/predict/<filename>', methods=['GET'])
def predict(filename):
    image_url = url_for('images', filename=filename)
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    image_data = load_and_prepare(image_path)
    print(hasattr(image_data, "_getexif"))
 
    # keras imports
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, GlobalAveragePooling2D
    from tensorflow.keras.layers import Dropout, InputLayer, BatchNormalization

    # instantiate the model
    NN = Sequential()
    NN.add(InputLayer(input_shape=(150, 150, 3)))
    # Conv block 1.  
    NN.add(Conv2D(filters=2, kernel_size=3, activation='elu', padding='same')) 
    NN.add(MaxPooling2D(2))
    NN.add(BatchNormalization())
    NN.add(Dropout(0.2))
    # Conv block 2
    NN.add(Conv2D(filters=4, kernel_size=3, activation='elu', padding='same'))
    NN.add(MaxPooling2D(2))
    NN.add(BatchNormalization())
    # Conv block 3
    NN.add(Conv2D(filters=8, kernel_size=3, activation='elu', padding='same'))
    NN.add(MaxPooling2D(2))
    NN.add(BatchNormalization())
    NN.add(Dropout(0.5))
    # Conv block 4
    NN.add(Conv2D(filters=12, kernel_size=3, activation='elu', padding='same'))
    NN.add(MaxPooling2D(2))
    NN.add(BatchNormalization())
    # Conv block 5
    NN.add(Conv2D(filters=24, kernel_size=3, activation='elu', padding='same'))
    NN.add(MaxPooling2D(2))
    NN.add(BatchNormalization())
    # Fully connected block - flattening followed by dense and output layers
    NN.add(Flatten())
    NN.add(Dense(4,  activation='elu'))
    NN.add(BatchNormalization())
    NN.add(Dropout(0.5))
    NN.add(Dense(2,activation='sigmoid'))  # 2 target classes, output layer
    
    # define the model and load the weights
    NEURAL_NET_MODEL_PATH = os.environ['NEURAL_NET_MODEL_PATH']
    NN.load_weights(NEURAL_NET_MODEL_PATH)

    # predict and generate bar graph
    predictions = NN.predict(image_data)[0]
    

    # imports for emotion detection
    from fer import FER
    import cv2
    
    detector = FER()
    image_data = image_data * 255
    image_data = image_data[0]
    image_data = image_data.astype('uint8')
    emotions = detector.detect_emotions(image_data)
    emotions_dict = emotions[0]['emotions']
    script, div = generate_barplot(emotions_dict)

    try:
        return render_template(
            'predict.html',
            plot_script=script,
            plot_div=div,
            image_url=image_url,
            message=predictions[1])

    except IndexError:
        return render_template(
            'predict_no_emotion.html',
            plot_script=script,
            plot_div=div,
            image_url=image_url,
            message='Oops! We could not capture your emotions with that photo. Please try again and make sure your face is clearly visible!')

@app.errorhandler(500)
def server_error(error):
    return render_template('error.html'), 500

@app.route('/images/<filename>', methods=['GET'])
def images(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))


