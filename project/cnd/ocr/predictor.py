# HERE YOUR PREDICTOR
from argus.model import load_model


class Predictor:
    def __init__(self, model_path, image_size, device="cuda"):
        self.model = load_model(model_path, device=device)
        self.ocr_image_size = image_size
        self.transform =  #TODO: prediction_transform

    def predict(self, images):
        #TODO: check for correct input type, you can receive one image [x,y,3] or batch [b,x,y,3]
        images = self.transform(images)
        pred = self.model.predict({"image": images})
        return pred
