import os
from flask import Flask, request, render_template
from base64 import b64encode
app = Flask(__name__)

from commons import get_tensor
from inference import get_fruit_name

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template('home.html', value='index')
    if request.method == 'POST':
        # check have file selected ?
        file = request.files['file']      
        if request.files['file'].filename == '':
            print('file not uploaded')
            return render_template('home.html', value='index')        
        image = file.read()
        #show image in view result base64 encode
        image_base_64_result = b64encode(image).decode("utf-8")

        #prediction, probs = get_fruit_name(image_bytes=image)
        top_probs, top_labels, top_fruits = get_fruit_name(image_bytes=image)
      

        return render_template('prediction.html', fruits=top_fruits, name=top_labels, probabilities=top_probs, imagebase64=image_base_64_result)



if __name__ == '__main__':
	app.run(debug=True)