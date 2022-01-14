from typing import Generator
import tensorflow as tf
from model import model_fn
from model.provider import ModelProvider


class DataGenerator(object):
    """
    Generator object that store a sample and generate it once while called.
    Used to feed a tensorflow estimator without knowing the whole data at
    build time.
    """

    def __init__(self) -> None:
        """ Default constructor. """
        self._current_data = None

    def update_data(self, data) -> None:
        """ Replace internal data. """
        self._current_data = data

    def __call__(self) -> Generator:
        """ Generation process. """
        buffer = self._current_data
        while buffer:
            yield buffer
            buffer = self._current_data


def create_estimator(params, MWF):
    """
    Initialize tensorflow estimator that will perform separation

    Params:
    - params: a dictionary of parameters for building the model

    Returns:
        a tensorflow estimator
    """
    # Load model.
    provider: ModelProvider = ModelProvider.default()
    params["model_dir"] = provider.get(params["model_dir"])
    params["MWF"] = MWF
    # Setup config
    session_config = tf.compat.v1.ConfigProto()
    session_config.gpu_options.per_process_gpu_memory_fraction = 0.7
    config = tf.estimator.RunConfig(session_config=session_config)
    # Setup estimator
    estimator = tf.estimator.Estimator(
        model_fn=model_fn, model_dir=params["model_dir"], params=params, config=config
    )
    return estimator