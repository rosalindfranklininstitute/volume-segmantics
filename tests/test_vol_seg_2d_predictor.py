import copy
import numpy as np
import pytest
import torch
import volume_segmantics.utilities.base_data_utils as utils
from volume_segmantics.model.model_2d import create_model_on_device
from volume_segmantics.model.operations.vol_seg_2d_predictor import VolSeg2dPredictor


@pytest.fixture()
def prediction_settings_single_channel(prediction_settings):
    settings = copy.deepcopy(prediction_settings)
    settings.use_2_5d_prediction = False
    return settings


@pytest.fixture()
def model_path_1channel(tmp_path, training_settings):
    model_struc_dict = copy.deepcopy(training_settings.model)
    model_struc_dict["type"] = utils.get_model_type(training_settings)
    model_struc_dict["in_channels"] = 1
    model_struc_dict["classes"] = 4
    model = create_model_on_device(0, model_struc_dict)
    model_dict = {
        "model_state_dict": model.state_dict(),
        "model_struc_dict": model_struc_dict,
        "label_codes": {},
    }
    path = tmp_path / "test_model_1ch.pytorch"
    torch.save(model_dict, path)
    yield path
    path.unlink()


@pytest.fixture()
def volseg_2d_predictor(model_path_1channel, prediction_settings_single_channel):
    return VolSeg2dPredictor(model_path_1channel, prediction_settings_single_channel)


class TestVolseg2DPredictor:
    @pytest.mark.gpu
    def test_2d_predictor_init(self, volseg_2d_predictor):
        assert isinstance(volseg_2d_predictor, VolSeg2dPredictor)
        assert isinstance(volseg_2d_predictor.model, torch.nn.Module)
        assert isinstance(volseg_2d_predictor.num_labels, int)
        assert isinstance(volseg_2d_predictor.label_codes, dict)

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_predict_single_axis(self, volseg_2d_predictor, rand_int_volume):
        result = volseg_2d_predictor._predict_single_axis(
            rand_int_volume, output_probs=False
        )
        labels = result[0]
        probs = result[1] if len(result) > 1 else None
        logits = result[2] if len(result) > 2 else None
        assert isinstance(labels, np.ndarray)
        assert labels.dtype == np.uint8
        assert probs is None
        assert logits is None

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_predict_single_axis_probs(self, volseg_2d_predictor, rand_int_volume):
        result = volseg_2d_predictor._predict_single_axis(
            rand_int_volume, output_probs=True
        )
        labels = result[0]
        probs = result[1] if len(result) > 1 else None
        logits = result[2] if len(result) > 2 else None
        assert isinstance(labels, np.ndarray)
        assert labels.dtype == np.uint8
        assert isinstance(probs, np.ndarray)
        assert probs.dtype == np.float16
        if logits is not None:
            assert isinstance(logits, np.ndarray)

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_predict_3_ways_max_probs(self, volseg_2d_predictor, rand_int_volume):
        labels, probs = volseg_2d_predictor._predict_3_ways_max_probs(rand_int_volume)
        assert isinstance(labels, np.ndarray)
        assert labels.dtype == np.uint8
        assert isinstance(probs, np.ndarray)
        assert probs.dtype == np.float16

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_predict_12_ways_max_probs(self, volseg_2d_predictor, rand_int_volume):
        labels, probs = volseg_2d_predictor._predict_12_ways_max_probs(rand_int_volume)
        assert isinstance(labels, np.ndarray)
        assert labels.dtype == np.uint8
        assert isinstance(probs, np.ndarray)
        assert probs.dtype == np.float16

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_predict_single_axis_one_hot(self, volseg_2d_predictor, rand_int_volume):
        counts = volseg_2d_predictor._predict_single_axis_to_one_hot(rand_int_volume)
        assert isinstance(counts, np.ndarray)
        assert counts.dtype == np.uint8
        assert counts.ndim == 4  # one hot output should be 4 dimensional

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_predict_3_axis_one_hot(self, volseg_2d_predictor, rand_int_volume):
        counts = volseg_2d_predictor._predict_3_ways_one_hot(rand_int_volume)
        assert isinstance(counts, np.ndarray)
        assert counts.dtype == np.uint8
        assert counts.ndim == 4  # one hot output should be 4 dimensional

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_predict_12_ways_one_hot(self, volseg_2d_predictor, rand_int_volume):
        counts = volseg_2d_predictor._predict_12_ways_one_hot(rand_int_volume)
        assert isinstance(counts, np.ndarray)
        assert counts.dtype == np.uint8
        assert counts.ndim == 4  # one hot output should be 4 dimensional
