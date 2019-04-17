from flask import Flask, request, jsonify, Response
from sklearn.externals import joblib
from iris_predictor import IrisPredictor
import numpy as np
def create_app(*args,**kwargs):
    
    app = Flask(__name__)
    model = IrisPredictor('model.pkl')

    @app.route("/predict_flower_type/", methods=['GET'])
    def getContents():
        
        content = request.data 
        print(request)

        x = [float(request.args.get('sepal_length')),
             float(request.args.get('sepal_width')),
             float(request.args.get('petal_length')),
             float(request.args.get('petal_width'))]
        
        return jsonify({"prediction":str(model.predict([x])[0])})

    @app.route('/health/')
    def health_check():
        return Response("", status = 200)

    @app.route('/ready/', methods= ["GET"])
    def readiness_check():
        return Response("", status = 200)

    return app


if __name__ == '__main__':
   
    app.run(host= '0.0.0.0', port= 5000, debug=True) 

