from estimator import DataGenerator, create_estimator
from stft import STFTBackend
import scipy.io.wavfile as wavf
from typing import Generator
import numpy as np
import tensorflow as tf
import os
from utils.configuration import load_configuration

filename = 'file.wav' # name of audio file to separate



stems_2 = 'spleeter:2stems'
stems_4 = 'spleeter:4stems'
stems_5 = 'spleeter:5stems'
stems_to_use = stems_2 # specify which variable to use, depending on how many stems is desired


class Separator(object):
    """ A wrapper class for performing separation. """

    def __init__(
            self,
            filename,
            stems,
            stft_backend: STFTBackend = STFTBackend.AUTO,
    ):
        """
        Default constructor.

        Parameters:
            filename:
                Name of audio file to be separated.
            stems:
                Indicates how many stems to split into (2, 4, or 5)
            
        """
        self._params = load_configuration(stems)
        # self._sample_rate = self._params["sample_rate"]
        self._MWF = False
        self._prediction_generator = None
        self._params["stft_backend"] = STFTBackend.resolve(stft_backend)
        self._data_generator = DataGenerator()
        self.file = filename
        self.audio = tf.io.read_file(self.file)  # tensorflow reads audio file

        # tf tensors are in the form: tf.Tensor<numpy array, shape, data type>
        self.waveform, self.sample_tensor = tf.audio.decode_wav(contents=self.audio)  # converts the file's contents
        # into a tuple containing a waveform tensor and a sample rate tensor

        self.sample_rate = self.sample_tensor.numpy()  # set sample_rate as the numpy array value from sample_tensor
        self.ndarray = tf.make_ndarray(tf.make_tensor_proto(self.waveform))  # extract numpy ndarray from waveform tensor
    def _get_prediction_generator(self):
        """
        Lazy loading access method for internal prediction generator
        returned by the predict method of a tensorflow estimator.

        Returns:
            Generator:
                Generator of prediction.
        """
        if self._prediction_generator is None:
            estimator = create_estimator(self._params, self._MWF)

            def get_dataset():
                return tf.data.Dataset.from_generator(
                    self._data_generator,
                    output_types={"waveform": tf.float32, "audio_id": tf.string},
                    output_shapes={"waveform": (None, 2), "audio_id": ()},
                )

            self._prediction_generator = estimator.predict(
                get_dataset, yield_single_examples=False
            )

        return self._prediction_generator

    def _separate_tensorflow(
            self, waveform: np.ndarray, audio_descriptor):
        """
        Performs source separation over the given waveform with tensorflow
        backend.

        Parameters:
            waveform (numpy.ndarray):
                Waveform to be separated (as a numpy array)
            audio_descriptor (AudioDescriptor):
                Note: audio_descriptor is irrelevant to my program but is kept to prevent errors

        Returns:
            Separated waveforms.
        """
        prediction_generator = self._get_prediction_generator()
        # NOTE: update data in generator before performing separation.
        self._data_generator.update_data(
            {"waveform": waveform, "audio_id": np.array(audio_descriptor)}
        )
        # NOTE: perform separation.
        prediction = next(prediction_generator)
        prediction.pop("audio_id")
        return prediction  # prediction is a dictionary in the form {'instrument': 'waveform data'}

    def output(self):
        stems = self._separate_tensorflow(self.ndarray, '1')  # run ndarray through separation function
        path = f'{os.getcwd()}/split_audio/{self.file[0:len(self.file) - 4]}' # to make new directory at cwd/split_audio
        # with name of file
        if(not os.path.isdir(path)): # if desired path does not exist
            print(('test'))
            os.mkdir(path) # make directory
        for instrument, data in stems.items():  # iterate over dictionary returned from separation function
            wavf.write(f'{path}/{instrument}.wav', self.sample_rate, data)  # for each instrument,
            # write that instrument's waveform data to corresponding file


separator = Separator(filename, stems_to_use)
separator.output()
