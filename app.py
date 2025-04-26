from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import pickle
import pandas as pd
import numpy as np
from flask import session
import uuid
import json
from datetime import datetime
from PIL import Image

import cv2
import os
import os


import tensorflow as tf

import os
import PIL
import cv2
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
foodlabel = pd.read_csv("foodlabeldataset.csv")
foodlabeldata = pd.read_csv("foodlabeldataset.csv")

foodlabel=foodlabel.drop(['Product Name','Ingredients','Reason'],axis=1)
Y=foodlabel['Healthy (1/0)'].values
X=foodlabel.drop(['Healthy (1/0)'],axis=1)
data=foodlabel.drop(['Healthy (1/0)'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(X.values, Y, test_size=0.2, random_state=42)
foodlabelinfo= ['Almonds',
 'Biscuits',
 'Chocolate',
 'Lays',
 'Maggi',
 'Oats',
 'Peanut Butter',
 'Soup']
# Define individual models
svm_clf = SVC(kernel='rbf', probability=True, random_state=42)
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)

# Create an ensemble model using Voting Classifier
ensemble_clf = VotingClassifier(estimators=[
    ('svm', svm_clf),
    ('rf', rf_clf),
    ('xgb', xgb_clf)
], voting='soft')

# Train the ensemble model
ensemble_clf.fit(X_train, y_train)
y_pred = ensemble_clf.predict(X_test)
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, BatchNormalization, GlobalAveragePooling2D, Dropout, Concatenate, Input
from tensorflow.keras.models import Model


model = load_model('foodlabel.h5')
#model.load_weights('foodlabel.h5')   
model.summary()
app = Flask(__name__)
app.config['SESSION_TYPE'] = 'memcached'
app.config['SECRET_KEY'] = 'super secret key'

CORS(app, resources={r"/*": {"origins": "*"}}) 

@app.route('/')
def hello():
    message= ''
    return render_template("home.html",message = message,name="Food Label Detect")

@app.route('/index')
def index():
    message= ''
    return render_template("home.html",message = message,name="Food Label Detect")

def preprocess(IMG_SAVE_PATH):
    dataset = []
   
    try:
                imgpath=PIL.Image.open(IMG_SAVE_PATH)
                imgpath=imgpath.convert('RGB')
                img = np.asarray(imgpath)
                img = cv2.resize(img, (331,331))
                img=img/255.
                dataset.append(img)
    except FileNotFoundError:
                print('Image file not found. Skipping...')
    return dataset
#model = tf.keras.models.load_model('foodlabel.h5')
#model.summary()

@app.route('/detect', methods=['POST'])
def detect():
    photo=request.files["photo"]
    photo_name = photo.filename
    protein = float(request.form.get('protein'))
    carbs = float(request.form.get('carbs'))
    fat = float(request.form.get('fat'))
    fiber = float(request.form.get('fiber'))
    sugar = float(request.form.get('sugar'))
    calories = float(request.form.get('calories'))
    calcium = float(request.form.get('calcium'))
    sodium = float(request.form.get('sodium'))
    photo.save("static/foodlabel/" + photo_name)
    traindata=preprocess("static/foodlabel/" + photo_name)
    xtest=np.array(traindata)
    filename="static/foodlabel/" + photo_name
    Y_pred = model.predict([xtest, xtest])
    Y_pred_classes = np.argmax(Y_pred,axis = 1) 
    #Y_pred_classes=0
    print(Y_pred_classes)
    message="Healthy"
    info=[]
    data=[protein,carbs,fat,fiber,sugar,calories,calcium,sodium]
    print(data)
    info.append(data)
    result=ensemble_clf.predict(info)
    print(result[0])
    print(result)
    foodlabeldata = pd.read_csv("foodlabeldataset.csv")
    filtered_df = foodlabeldata[(foodlabeldata['Product Name'] == foodlabelinfo[Y_pred_classes[0]]) & (foodlabeldata['Healthy (1/0)'] == result[0]) ]
    print(filtered_df)
    reason=filtered_df['Reason'].values.tolist()[0]
    health_check="Healthy"
    if(result==1):
        health_check="Healthy"
    else:
        health_check = "Not Healthy"


 
    

    return render_template("result.html",message = foodlabelinfo[Y_pred_classes[0]],reason=reason, image=filename , result=health_check)
  
 


@app.route('/home')
def home():
    message= ''
    return render_template("home.html",message = message,name="Food Label Detect")



if __name__ == '__main__':
    app.run(debug=True)
