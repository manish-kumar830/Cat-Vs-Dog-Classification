import streamlit as st
import numpy as np
from PIL import Image
import tensorflow_hub as hub
# from keras.models import load_model
from tensorflow.keras.models import load_model
import cv2


# Model saved with Keras model.save()
MODEL_PATH = 'models/model.h5'

# Load your trained model
def loading_model():
    model = load_model((MODEL_PATH),custom_objects={'KerasLayer':hub.KerasLayer})
    return model

model = loading_model()




st.text("""""")
image = Image.open("static/title_image.jpg")
st.image(
	        image,
	        use_column_width=True,
	    )
st.title("""
Cat 🐱 Or Dog 🐶 Recognizer
	""")



def model_predict(image_path, model):
    input_image = cv2.imread(image_path)

    input_image_resize = cv2.resize(input_image, (224,224))

    input_image_scaled = input_image_resize/255

    image_reshaped = np.reshape(input_image_scaled, [1,224,224,3])

    input_prediction = model.predict(image_reshaped)

    input_pred_label = np.argmax(input_prediction)

    return input_pred_label

def main():
    file = st.file_uploader("Upload Image",type=['jpg','png','jpeg'])
    if file is not None:
        st.image(file,use_column_width=True)
        save_image_path = 'uploads/'+file.name
        with open(save_image_path,'wb') as f:
            f.write(file.getbuffer())

        if st.button("predict"):
            result = model_predict(save_image_path,model)
            if result==0:
                st.error("Prediction : Cat")
            else:
                st.error("Prediction : Dog")
    


if __name__ == '__main__':
    main()
    
