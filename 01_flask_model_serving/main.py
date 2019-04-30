import argparse
import logging
import json
import time

import numpy as np
from sklearn.externals import joblib


from flask import Flask
from flask import make_response
from flask import request
from flask import current_app

from flask_classful import FlaskView, route


def output_json(data, code, headers=None):
    """
    This method is used to serialize the python dict to a json
    :param data:
    :param code:
    :param headers:
    :return:
    """
    content_type = 'application/json'
    dumped = json.dumps(data)
    if headers:
        headers.update({'Content-Type': content_type})
    else:
        headers = {'Content-Type': content_type}
    response = make_response(dumped, code, headers)
    return response


def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()


class PredictView(FlaskView):
    representations = {'application/json': output_json}

    def post(self):
        #print("Got predict")
        #print(request.method)
        model = current_app.config["model"]
        if model is None:
            return {'error': "model is not loaded"}
        #print(request)
        #print(request.content_type)
        content = request.get_json(force=True)
        if content is None:
            return {'error': 'content of request is None'}, 404

        # print("Content is: {}".format(content))
        if "sample" not in content:
            return {'error': "sample key is not found in content"}

        sample = content["sample"]

        start = time.time()
        np_sample = np.asarray(sample).reshape(-1, len(sample))
        #print("Sample: {}".format(np_sample))

        np_pred = model.predict(np_sample)
        np_pred_prob = model.predict_proba(np_sample)[0]

        list_pred = list(np_pred)
        list_pred_prob = list(np_pred_prob)

        list_pred = list(map(float, list_pred))
        list_pred_prob = list(map(float, list_pred_prob))
        end = time.time()
        #print("prediction: {}".format(list_pred))

        total_time = end - start
        return {'prediction': list_pred, "prediction_probability": list_pred_prob, "time": total_time}

    def shutdown(self):
        shutdown_server()
        return 'Server shutting down...'


def serv_predictions(port, rest_endpoint, input_model):
    """
    Main function running the prediction server
    :param port:
    :param rest_endpoint:
    :input_model: Path to model file to load
    :return:
    """

    print("Port:      {}".format(port))
    print("Endpoint:  {}".format(rest_endpoint))
    print("Model:     {}".format(input_model))
    print("================")

    # Loading the model
    loaded_model = joblib.load(input_model)

    app = Flask(__name__)
    app.config["model"] = loaded_model
    print("Starting..")
    PredictView.register(app, route_base=rest_endpoint)

    app.run(host="0.0.0.0", port=port)
    print("App is done ... exiting")


if __name__ == '__main__':

    log_levels = {"debug": logging.DEBUG, "info": logging.INFO, "warning": logging.WARNING, "error": logging.ERROR}

    parser = argparse.ArgumentParser()
    parser.add_argument("port", type=int, help="Port to listen on")
    parser.add_argument("model", help="Path of model to load")
    parser.add_argument("--rest-endpoint", default="/predict/", help="Endpoint to register")
    parser.add_argument("--log-level", choices=log_levels.keys(), default="info", help="Logging level")
    options = parser.parse_args()

    logging.basicConfig(format='%(asctime)-15s %(levelname)s [%(module)s:%(lineno)d]:  %(message)s')
    logging.getLogger('mlpiper').setLevel(log_levels[options.log_level])

    serv_predictions(options.port, options.rest_endpoint, options.model)

