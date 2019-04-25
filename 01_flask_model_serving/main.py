import argparse
import logging
import logging
import numpy as np
import pickle

from flask import Flask
from flask_restplus import Resource, Api

app = Flask(__name__)
api = Api(app)


class SklearnRESTfulServing(Resource):
    JSON_KEY_NAME = "prediction_vector"

    def __init__(self):
        self._logger = logging.getLogger(SklearnRESTfulServing.__name__)
        self._model = None
        self._verbose = self._logger.isEnabledFor(logging.DEBUG)

    def load_model(self, model_path):
        self._logger.info("Model is loading, path: {}".format(model_path))
        with open(model_path, "rb") as f:
            self._model = pickle.load(f)
            if self._verbose:
                self._logger.debug("Un-pickled model: {}".format(self._model))
            self._logger.info("Model loaded successfully!")

    @api.route('/predict')
    def predict(self, url_params, form_params):
        if SklearnRESTfulServing.JSON_KEY_NAME not in form_params:
            raise Exception("Unexpected json format for prediction! Missing '{}' key!"
                            .format(SklearnRESTfulServing.JSON_KEY_NAME))

        if self._verbose:
            self._logger.debug("predict, url_params: {}, form_params: {}".format(url_params, form_params))

        if not self._model:
            return 404, {"error": "Model not loaded yet!"}
        else:
            if self._verbose:
                self._logger.debug("type<form_params>: {}\n{}".format(type(form_params), form_params))
                two_dim_array = np.array([form_params[SklearnRESTfulServing.JSON_KEY_NAME]])
                self._logger.debug("type(two_dim_array): {}\n{}".format(type(two_dim_array), two_dim_array))
                result = self._model.predict(two_dim_array)
                self._logger.debug("result: {}, type: {}".format(result[0], type(result[0])))
            else:
                two_dim_array = np.array([form_params[SklearnRESTfulServing.JSON_KEY_NAME]])
                result = self._model.predict(two_dim_array)

            return 200, {"result": result[0]}


if __name__ == '__main__':

    log_levels = {"debug": logging.DEBUG, "info": logging.INFO, "warning": logging.WARNING, "error": logging.ERROR}

    parser = argparse.ArgumentParser()
    parser.add_argument("port", type=int, help="Port to listen on")
    parser.add_argument("model", help="Path of model to load")
    parser.add_argument("--log-level", choices=log_levels.keys(), default="info", help="Logging level")
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)-15s %(levelname)s [%(module)s:%(lineno)d]:  %(message)s')
    logging.getLogger('mlpiper').setLevel(log_levels[args.log_level])

    app.run(debug=True)





