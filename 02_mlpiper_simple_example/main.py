import logging
from parallelm.components.restful.flask_route import FlaskRoute
from parallelm.components.restful_component import RESTfulComponent


class SimpleRESTfulServing(RESTfulComponent):
    def __init__(self, engine):
        super(SimpleRESTfulServing, self).__init__(engine)
        self._last_model = None
        self._content = None
        self._counter = 0

    def _configure(self, params):
        pass

    def load_model(self, model_path, stream, version):
        self._logger.info("Model is reloading, path: {}".format(model_path))
        self._last_model = model_path

        with open(model_path, 'r') as f:
            self._content = f.read()

    @FlaskRoute('/v1/predict')
    def predict_v1(self, url_params, form_params):
        if self._logger.isEnabledFor(logging.DEBUG):
            self._logger.debug("predict_v1")
        status_code = 200
        return (status_code, self._build_response('jon2', 44, url_params))

    @FlaskRoute('/v2/predict')
    def predict_v2(self, url_params, form_params):
        if self._logger.isEnabledFor(logging.DEBUG):
            self._logger.debug("predict_v2")
        status_code = 200
        return (status_code, self._build_response('jon2', 64, url_params))

    @FlaskRoute('/exception')
    def predic_exception(self, url_params, form_params):
        raise Exception("This is a dummy exception from the MLApp!")

    def _build_response(self, user, age, params):
        self._counter += 1
        return {
            'user': user,
            'age': age,
            'last_model': self._last_model,
            'content': self._content,
            'counter': self._counter,
            'params': params
        }


if __name__ == '__main__':
    SimpleRESTfulServing.run(port=9999, model_path='/tmp/dummy-model.txt')
