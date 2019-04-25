import logging
import numpy as np
import pickle

from parallelm.components.restful.flask_route import FlaskRoute
from parallelm.components.restful_component import RESTfulComponent


class SklearnRESTfulServing(RESTfulComponent):
    JSON_KEY_NAME = "prediction_vector"

    def __init__(self, engine):
        super(SklearnRESTfulServing, self).__init__(engine)
        self._classifier = None
        self._verbose = self._logger.isEnabledFor(logging.DEBUG)

    def _configure(self, params):
        pass

    def load_model(self, model_path, stream, version):
        self._logger.info("Model is reloading, path: {}".format(model_path))
        with open(model_path, "rb") as f:
            self._classifier = pickle.load(f)
            if self._verbose:
                self._logger.debug("Un-pickled model: {}".format(self._classifier))
            self._logger.info("Model loaded successfully!")

    @FlaskRoute('/predict')
    def predict(self, url_params, form_params):
        if SklearnRESTfulServing.JSON_KEY_NAME not in form_params:
            raise Exception("Unexpected json format for prediction! Missing '{}' key!"
                            .format(SklearnRESTfulServing.JSON_KEY_NAME))

        if self._verbose:
            self._logger.debug("predict, url_params: {}, form_params: {}".format(url_params, form_params))

        if not self._classifier:
            return (404, {"error": "Model not loaded yet!"})
        else:
            if self._verbose:
                self._logger.debug("type<form_params>: {}\n{}".format(type(form_params), form_params))
                two_dim_array = np.array([form_params[SklearnRESTfulServing.JSON_KEY_NAME]])
                self._logger.debug("type(two_dim_array): {}\n{}".format(type(two_dim_array), two_dim_array))
                result = self._classifier.predict(two_dim_array)
                self._logger.debug("result: {}, type: {}".format(result[0], type(result[0])))
            else:
                two_dim_array = np.array([form_params[SklearnRESTfulServing.JSON_KEY_NAME]])
                result = self._classifier.predict(two_dim_array)

            return (200, {"result": result[0]})


if __name__ == '__main__':
    import argparse

    log_levels = {"debug": logging.DEBUG, "info": logging.INFO, "warning": logging.WARNING, "error": logging.ERROR}

    parser = argparse.ArgumentParser()
    parser.add_argument("port", type=int, help="Port to listen on")
    parser.add_argument("input_model", help="Path of input model to create")
    parser.add_argument("--log_level", choices=log_levels.keys(), default="info", help="Logging level")
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)-15s %(levelname)s [%(module)s:%(lineno)d]:  %(message)s')
    logging.getLogger('parallelm').setLevel(log_levels[args.log_level])

    SklearnRESTfulServing.run(port=args.port, model_path=args.input_model)
