import numpy as np
import pytest
import torch
from torch.nn import DataParallel

from volume_segmantics.model.operations.vol_seg_2d_predictor import VolSeg2dPredictor


@pytest.fixture()
def volseg_2d_predictor(model_path, prediction_settings):
    return VolSeg2dPredictor(model_path, prediction_settings)


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
        labels, probs, logits = volseg_2d_predictor._predict_single_axis(
            rand_int_volume, output_probs=False
        )
        assert isinstance(labels, np.ndarray)
        assert labels.dtype == np.uint8
        assert probs is None
        assert logits is None

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_predict_single_axis_probs(self, volseg_2d_predictor, rand_int_volume):
        labels, probs, logits = volseg_2d_predictor._predict_single_axis(
            rand_int_volume, output_probs=True
        )
        assert isinstance(labels, np.ndarray)
        assert labels.dtype == np.uint8
        assert isinstance(probs, np.ndarray)
        assert probs.dtype == np.float16
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

    @pytest.mark.gpu
    def test_predict_3_ways_output_shape_matches_volume(self, volseg_2d_predictor, rand_int_volume):
        """_predict_3_ways_max_probs returns labels and probs with same shape as input volume."""
        labels, probs = volseg_2d_predictor._predict_3_ways_max_probs(rand_int_volume)
        assert labels.shape == rand_int_volume.shape
        assert probs.shape == rand_int_volume.shape
        assert np.all((labels >= 0) & (labels < volseg_2d_predictor.num_labels))
        assert np.all((probs >= 0) & (probs <= 1.0))

    @pytest.mark.gpu
    def test_predict_12_ways_output_shape_matches_volume(self, volseg_2d_predictor, rand_int_volume):
        """_predict_12_ways_max_probs returns labels and probs with same shape as input volume."""
        labels, probs = volseg_2d_predictor._predict_12_ways_max_probs(rand_int_volume)
        assert labels.shape == rand_int_volume.shape
        assert probs.shape == rand_int_volume.shape
        assert np.all((labels >= 0) & (labels < volseg_2d_predictor.num_labels))

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_prediction_estimate_entropy_medium_returns_shapes(self, volseg_2d_predictor, rand_int_volume):
        """_prediction_estimate_entropy with quality medium returns labels, probs, entropy, counts."""
        volseg_2d_predictor.settings.quality = "medium"
        labels, probs, entropy_matrix, counts = volseg_2d_predictor._prediction_estimate_entropy(
            rand_int_volume
        )
        assert labels.shape == rand_int_volume.shape
        assert probs.shape == rand_int_volume.shape
        assert entropy_matrix.shape == rand_int_volume.shape
        assert counts.shape == (volseg_2d_predictor.num_labels,) + rand_int_volume.shape
        assert labels.dtype == np.uint8
        assert np.all((labels >= 0) & (labels < volseg_2d_predictor.num_labels))

    @pytest.mark.gpu
    def test_prediction_estimate_entropy_low_raises(self, volseg_2d_predictor, rand_int_volume):
        """_prediction_estimate_entropy with quality low raises ValueError."""
        volseg_2d_predictor.settings.quality = "low"
        with pytest.raises(ValueError, match="minimum prediction quality of medium"):
            volseg_2d_predictor._prediction_estimate_entropy(rand_int_volume)

    @pytest.mark.gpu
    def test_convert_labels_map_to_count_synthetic(self, volseg_2d_predictor):
        """_convert_labels_map_to_count: hand-check counts for a tiny label map."""
        # Labels 0, 1, 2 in a 2x2 grid
        labels_vol = np.array([[[0, 1], [1, 2]]], dtype=np.uint8)
        counts_matrix, label_sorted = volseg_2d_predictor._convert_labels_map_to_count(labels_vol)
        assert counts_matrix.shape[0] == len(label_sorted)
        assert counts_matrix.shape[1:] == labels_vol.shape
        # Each label index should sum to number of pixels with that label
        assert np.sum(counts_matrix[0]) == np.sum(labels_vol == label_sorted[0])
        assert np.sum(counts_matrix[1]) == np.sum(labels_vol == label_sorted[1])
        assert np.sum(counts_matrix[2]) == np.sum(labels_vol == label_sorted[2])
