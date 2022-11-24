import gradio as gr
import numpy as np
from PIL import Image
import requests

import hopsworks
import joblib

project = hopsworks.login()
fs = project.get_feature_store()


mr = project.get_model_registry()
model = mr.get_model("titanic_modal", version=1)
model_dir = model.download()
model = joblib.load(model_dir + "/titanic_model.pkl")


def titanic(Pclass, Sex, Age, SibSp, Parch, Embarked):
    input_list = []
    
    input_list.append(Pclass)
    input_list.append(Sex)
    input_list.append(Age)
    input_list.append(SibSp)
    input_list.append(Parch)
    input_list.append(Embarked)
    # 'res' is a list of predictions returned as the label.
    res = model.predict(np.asarray(input_list).reshape(1, -1)) 
    if res[0]==0:
        link ="https://github.com/JeetNimbhorkar/TitanicLab1/raw/d9482baa7cbe47d0a8d5dcbe93e1ce7c0b2538a2/didnotsurvive.png"
    else:
        link = "https://github.com/JeetNimbhorkar/TitanicLab1/raw/d9482baa7cbe47d0a8d5dcbe93e1ce7c0b2538a2/survived.png"
    # We add '[0]' to the result of the transformed 'res', because 'res' is a list, and we only want 
    # the first element.
    #flower_url = "https://raw.githubusercontent.com/featurestoreorg/serverless-ml-course/main/src/01-module/assets/" + pred + ".png"
    titanic_url=link
    img = Image.open(requests.get(titanic_url, stream=True).raw)            
    return img
        
demo = gr.Interface(
    fn=titanic,
    title="Titanic survival Predictive Analytics",
    description="Enter passanger details to predict survival in Titanic",
    allow_flagging="never",
    inputs=[
        gr.inputs.Number(default=1.0, label="Pclass (Enter 1,2 or 3)"),
        gr.inputs.Number(default=1.0, label="Sex (0 for Male, 1 for Female)"),
        gr.inputs.Number(default=1.0, label="Age"),
        gr.inputs.Number(default=1.0, label="SibSp (Enter 0,1,2,3,4,5 or 8)"),
        gr.inputs.Number(default=1.0, label="Parch (Enter 0,1,2,3,4,5 or 6)"),
        gr.inputs.Number(default=1.0, label="Embarked (Enter 0 for C, 1 for Q and 2 for S)")
        ],
    outputs=gr.Image(type="pil"))

demo.launch()

