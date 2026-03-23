import numpy as np
from volume_segmantics.data import TrainingDataSlicer
import volume_segmantics.utilities.base_data_utils as utils
import pytest
from skimage import io


@pytest.fixture()
def training_data_slicer(data_vol, label_vol, training_settings, request):
    data_vol = request.getfixturevalue(data_vol)
    label_vol = request.getfixturevalue(label_vol)
    return TrainingDataSlicer(data_vol, label_vol, training_settings)


class TestTrainingDataSlicer:
    @pytest.mark.parametrize(
        "data_vol, label_vol",
        [
            ("rand_int_volume", "rand_label_volume"),
            ("rand_int_volume", "rand_label_tiff_path"),
            ("rand_int_volume", "rand_label_hdf5_path"),
            ("rand_int_hdf5_path", "rand_label_volume"),
            ("rand_int_hdf5_path", "rand_label_tiff_path"),
            ("rand_int_hdf5_path", "rand_label_hdf5_path"),
            ("rand_int_tiff_path", "rand_label_hdf5_path"),
            ("rand_int_tiff_path", "rand_label_tiff_path"),
            ("rand_int_tiff_path", "rand_label_volume"),
        ],
    )
    def test_training_data_slicer_init(self, training_data_slicer):
        assert isinstance(training_data_slicer, TrainingDataSlicer)
        assert isinstance(training_data_slicer.seg_vol, np.ndarray)
        assert training_data_slicer.codes is not None

    def test_training_data_slicer_fix_labels(
        self, rand_int_volume, rand_label_volume_no_zeros, training_settings
    ):
        assert np.unique(rand_label_volume_no_zeros)[0] != 0
        slicer = TrainingDataSlicer(
            rand_int_volume, rand_label_volume_no_zeros, training_settings
        )
        assert np.unique(slicer.seg_vol)[0] == 0
        # Check the label values are sequential
        values = np.unique(slicer.seg_vol)
        assert np.where(np.diff(values) != 1)[0].size == 0

    def test_training_data_slicer_output_data(
        self, rand_int_volume, rand_label_volume, training_settings, empty_dir
    ):
        im_dir_path = empty_dir / "im_out"
        slicer = TrainingDataSlicer(
            rand_int_volume, rand_label_volume, training_settings
        )
        slicer.output_data_slices(im_dir_path, "data")
        file_list = list(im_dir_path.glob("*.png"))
        assert len(file_list) != 0
        img = io.imread(file_list[0])
        assert isinstance(img, np.ndarray)
        assert np.issubdtype(img.dtype, np.integer)

    def test_training_data_slicer_output_data_all_axes(
        self, rand_int_volume, rand_label_volume, training_settings, empty_dir
    ):
        im_dir_path = empty_dir / "im_out"
        if hasattr(training_settings, "training_axis"):
            delattr(training_settings, "training_axis")
        slicer = TrainingDataSlicer(
            rand_int_volume, rand_label_volume, training_settings
        )
        slicer.output_data_slices(im_dir_path, "data")
        file_list = list(im_dir_path.glob("*.png"))
        assert len(file_list) != 0
        assert len(file_list) == sum(rand_int_volume.shape)

    def test_training_data_slicer_output_data_single_axis(
        self, rand_int_volume, rand_label_volume, training_settings, empty_dir
    ):
        im_dir_path = empty_dir / "im_out"
        training_settings.training_axes = "y"
        slicer = TrainingDataSlicer(
            rand_int_volume, rand_label_volume, training_settings
        )
        slicer.output_data_slices(im_dir_path, "data")
        file_list = list(im_dir_path.glob("*.png"))
        assert len(file_list) != 0
        assert len(file_list) == rand_int_volume.shape[1]

    def test_training_data_slicer_output_labels(
        self, rand_int_volume, rand_label_volume, training_settings, empty_dir
    ):
        label_dir_path = empty_dir / "label_out"
        slicer = TrainingDataSlicer(
            rand_int_volume, rand_label_volume, training_settings
        )
        slicer.output_label_slices(label_dir_path, "seg")
        file_list = list(label_dir_path.glob("*.png"))
        assert len(file_list) != 0
        img = io.imread(file_list[0])
        assert isinstance(img, np.ndarray)
        assert np.issubdtype(img.dtype, np.integer)

    def test_training_data_slicer_output_labels_all_axes(
        self, rand_int_volume, rand_label_volume, training_settings, empty_dir
    ):
        label_dir_path = empty_dir / "label_out"
        if hasattr(training_settings, "training_axis"):
            delattr(training_settings, "training_axis")
        slicer = TrainingDataSlicer(
            rand_int_volume, rand_label_volume, training_settings
        )
        slicer.output_label_slices(label_dir_path, "seg")
        file_list = list(label_dir_path.glob("*.png"))
        assert len(file_list) != 0
        assert len(file_list) == sum(rand_int_volume.shape)

    def test_training_data_slicer_output_labels_single_axis(
        self, rand_int_volume, rand_label_volume, training_settings, empty_dir
    ):
        label_dir_path = empty_dir / "label_out"
        training_settings.training_axes = "x"
        slicer = TrainingDataSlicer(
            rand_int_volume, rand_label_volume, training_settings
        )
        slicer.output_label_slices(label_dir_path, "seg")
        file_list = list(label_dir_path.glob("*.png"))
        assert len(file_list) != 0
        assert len(file_list) == rand_int_volume.shape[2]

    def test_training_data_slicer_output_binary_labels(
        self, rand_int_volume, rand_binary_label_volume, training_settings, empty_dir
    ):
        label_dir_path = empty_dir / "binary_label_out"
        slicer = TrainingDataSlicer(
            rand_int_volume, rand_binary_label_volume, training_settings
        )
        slicer.output_label_slices(label_dir_path, "seg")
        file_list = list(label_dir_path.glob("*.png"))
        for fn in file_list:
            img = io.imread(fn)
            assert isinstance(img, np.ndarray)
            assert np.issubdtype(img.dtype, np.integer)
            assert np.array_equal(np.unique(img), np.array([0, 1]))

    def test_training_data_slicer_clean_up(
        self, rand_int_volume, rand_label_volume, training_settings, empty_dir
    ):
        im_dir_path = empty_dir / "temp_im_out"
        label_dir_path = empty_dir / "temp_label_out"
        slicer = TrainingDataSlicer(
            rand_int_volume, rand_label_volume, training_settings
        )
        slicer.output_data_slices(im_dir_path, "data")
        slicer.output_label_slices(label_dir_path, "seg")
        im_file_list = list(im_dir_path.glob("*.png"))
        label_file_list = list(label_dir_path.glob("*.png"))
        assert len(im_file_list) != 0
        assert len(label_file_list) != 0
        slicer.clean_up_slices()
        assert not im_dir_path.exists()
        assert not label_dir_path.exists()

    def test_training_data_slicer_2_5d_invalid_num_slices_even(
        self, rand_int_volume, rand_label_volume, training_settings
    ):
        training_settings.use_2_5d_slicing = True
        training_settings.num_slices = 4  # must be odd
        with pytest.raises(ValueError):
            TrainingDataSlicer(rand_int_volume, rand_label_volume, training_settings)

    def test_training_data_slicer_2_5d_invalid_num_slices_too_small(
        self, rand_int_volume, rand_label_volume, training_settings
    ):
        training_settings.use_2_5d_slicing = True
        training_settings.num_slices = 1  # must be >= 3
        with pytest.raises(ValueError):
            TrainingDataSlicer(rand_int_volume, rand_label_volume, training_settings)

    def test_training_data_slicer_2_5d_png_too_many_channels_raises(
        self, rand_int_volume, rand_label_volume, training_settings
    ):
        training_settings.use_2_5d_slicing = True
        training_settings.num_slices = 5
        training_settings.slice_file_format = "png"
        with pytest.raises(ValueError):
            TrainingDataSlicer(rand_int_volume, rand_label_volume, training_settings)

    def test_training_data_slicer_2_5d_output_shape_and_channels(
        self, rand_int_volume, rand_label_volume, training_settings, empty_dir
    ):
        training_settings.use_2_5d_slicing = True
        training_settings.num_slices = 3
        training_settings.slice_file_format = "tiff"
        im_dir_path = empty_dir / "im_out_25d"
        slicer = TrainingDataSlicer(
            rand_int_volume, rand_label_volume, training_settings
        )
        # Use private helper to avoid disk I/O complexity for TIFF
        axis = "z"
        index = rand_int_volume.shape[0] // 2
        mc_slice = slicer._create_2_5d_slice(rand_int_volume, axis, index)
        assert mc_slice.shape[2] == training_settings.num_slices

    #  Init and constructor edge cases (increase coverage) 

    def test_training_data_slicer_init_requires_settings(
        self, rand_int_volume, rand_label_volume
    ):
        """__init__ with settings=None raises ValueError."""
        with pytest.raises(ValueError, match="settings parameter is required"):
            TrainingDataSlicer(rand_int_volume, rand_label_volume, None)

    def test_training_data_slicer_init_unlabeled(
        self, rand_int_volume, training_settings
    ):
        """__init__ with label_vol=None: unlabeled mode, seg_vol None, codes empty."""
        slicer = TrainingDataSlicer(
            rand_int_volume, None, training_settings
        )
        assert slicer.seg_vol is None
        assert slicer.num_seg_classes == 0
        assert slicer.codes == []
        assert slicer.has_labels is False

    def test_training_data_slicer_label_type_stored(
        self, rand_int_volume, rand_label_volume, training_settings
    ):
        """label_type is stored and used (e.g. for task2)."""
        slicer = TrainingDataSlicer(
            rand_int_volume, rand_label_volume, training_settings,
            label_type="task2",
        )
        assert slicer.label_type == "task2"

    # _create_2_5d_slice branches (axis, borders, invalid) 

    def test_create_2_5d_slice_axis_y(
        self, rand_int_volume, rand_label_volume, training_settings
    ):
        """_create_2_5d_slice along axis 'y' returns correct shape."""
        training_settings.use_2_5d_slicing = True
        training_settings.num_slices = 3
        slicer = TrainingDataSlicer(
            rand_int_volume, rand_label_volume, training_settings
        )
        axis, index = "y", rand_int_volume.shape[1] // 2
        mc_slice = slicer._create_2_5d_slice(rand_int_volume, axis, index)
        assert mc_slice.shape[2] == 3
        assert mc_slice.ndim == 3

    def test_create_2_5d_slice_axis_x(
        self, rand_int_volume, rand_label_volume, training_settings
    ):
        """_create_2_5d_slice along axis 'x' returns correct shape."""
        training_settings.use_2_5d_slicing = True
        training_settings.num_slices = 3
        slicer = TrainingDataSlicer(
            rand_int_volume, rand_label_volume, training_settings
        )
        axis, index = "x", rand_int_volume.shape[2] // 2
        mc_slice = slicer._create_2_5d_slice(rand_int_volume, axis, index)
        assert mc_slice.shape[2] == 3

    def test_create_2_5d_slice_border_index_zero(
        self, rand_int_volume, rand_label_volume, training_settings
    ):
        """_create_2_5d_slice at index 0 duplicates edge (no out-of-range)."""
        training_settings.use_2_5d_slicing = True
        training_settings.num_slices = 3
        slicer = TrainingDataSlicer(
            rand_int_volume, rand_label_volume, training_settings
        )
        mc_slice = slicer._create_2_5d_slice(rand_int_volume, "z", 0)
        assert mc_slice.shape[2] == 3
        assert not np.any(np.isnan(mc_slice))

    def test_create_2_5d_slice_border_index_last(
        self, rand_int_volume, rand_label_volume, training_settings
    ):
        """_create_2_5d_slice at last index duplicates edge."""
        training_settings.use_2_5d_slicing = True
        training_settings.num_slices = 3
        slicer = TrainingDataSlicer(
            rand_int_volume, rand_label_volume, training_settings
        )
        last_idx = rand_int_volume.shape[0] - 1
        mc_slice = slicer._create_2_5d_slice(rand_int_volume, "z", last_idx)
        assert mc_slice.shape[2] == 3

    def test_create_2_5d_slice_invalid_axis_raises(
        self, rand_int_volume, rand_label_volume, training_settings
    ):
        """_create_2_5d_slice with invalid axis raises ValueError."""
        training_settings.use_2_5d_slicing = True
        training_settings.num_slices = 3
        slicer = TrainingDataSlicer(
            rand_int_volume, rand_label_volume, training_settings
        )
        with pytest.raises(ValueError, match="Invalid axis"):
            slicer._create_2_5d_slice(rand_int_volume, "w", 0)

    # 2.5D output to disk (tiff) and _output_im format branch 

    def test_training_data_slicer_2_5d_output_tiff_to_disk(
        self, rand_int_volume, rand_label_volume, training_settings, empty_dir
    ):
        """output_data_slices in 2.5D mode writes .tiff files."""
        training_settings.use_2_5d_slicing = True
        training_settings.num_slices = 3
        training_settings.slice_file_format = "tiff"
        training_settings.training_axes = "z"
        im_dir_path = empty_dir / "im_25d_tiff"
        slicer = TrainingDataSlicer(
            rand_int_volume, rand_label_volume, training_settings
        )
        slicer.output_data_slices(im_dir_path, "data")
        tiff_list = list(im_dir_path.glob("*.tiff")) + list(im_dir_path.glob("*.tif"))
        assert len(tiff_list) >= 1
        img = io.imread(tiff_list[0])
        assert img.ndim == 3
        assert img.shape[-1] == 3

    def test_training_data_slicer_2_5d_unsupported_format_raises(
        self, rand_int_volume, rand_label_volume, training_settings, empty_dir
    ):
        """2.5D with unsupported slice_file_format raises ValueError when writing."""
        training_settings.use_2_5d_slicing = True
        training_settings.num_slices = 3
        training_settings.slice_file_format = "jpg"
        training_settings.training_axes = "z"
        im_dir_path = empty_dir / "im_25d_jpg"
        slicer = TrainingDataSlicer(
            rand_int_volume, rand_label_volume, training_settings
        )
        with pytest.raises(ValueError, match="Unsupported file format"):
            slicer.output_data_slices(im_dir_path, "data")

    # Labels always 2D when use_2_5d_slicing (branch in _output_slices_to_disk) 

    def test_training_data_slicer_2_5d_labels_still_2d(
        self, rand_int_volume, rand_label_volume, training_settings, empty_dir
    ):
        """With use_2_5d_slicing=True, label slices are still 2D (single channel)."""
        training_settings.use_2_5d_slicing = True
        training_settings.num_slices = 3
        training_settings.training_axes = "z"
        label_dir_path = empty_dir / "label_25d"
        slicer = TrainingDataSlicer(
            rand_int_volume, rand_label_volume, training_settings
        )
        slicer.output_label_slices(label_dir_path, "seg")
        png_list = list(label_dir_path.glob("*.png"))
        assert len(png_list) >= 1
        img = io.imread(png_list[0])
        assert img.ndim == 2

    # Multilabel: per-class labels preserved (not binarized in _output_im) 

    def test_training_data_slicer_multilabel_output_preserves_classes(
        self, rand_int_volume, training_settings, empty_dir
    ):
        """With 3+ classes, output label slices keep multiple class values (multilabel path)."""
        label_vol = np.random.randint(0, 4, size=rand_int_volume.shape)
        training_settings.training_axes = "z"
        label_dir_path = empty_dir / "multilabel_out"
        slicer = TrainingDataSlicer(
            rand_int_volume, label_vol, training_settings
        )
        assert slicer.multilabel is True
        slicer.output_label_slices(label_dir_path, "seg")
        files = list(label_dir_path.glob("*.png"))
        assert len(files) >= 1
        uniq = np.unique(io.imread(files[0]))
        assert len(uniq) >= 2

    # clean_up_slices with task2/task3 dirs 

    def test_training_data_slicer_clean_up_task2_dir(
        self, rand_int_volume, rand_label_volume, training_settings, empty_dir
    ):
        """clean_up_slices removes task2_im_out_dir if set (deletes images then rmdir)."""
        im_dir_path = empty_dir / "data_out"
        label_dir_path = empty_dir / "label_out"
        task2_dir = empty_dir / "task2_out"
        task2_dir.mkdir(exist_ok=True)
        io.imsave(str(task2_dir / "dummy.png"), np.zeros((4, 4), dtype=np.uint8), check_contrast=False)
        slicer = TrainingDataSlicer(
            rand_int_volume, rand_label_volume, training_settings
        )
        slicer.output_data_slices(im_dir_path, "data")
        slicer.output_label_slices(label_dir_path, "seg")
        slicer.task2_im_out_dir = task2_dir
        slicer.clean_up_slices()
        assert not im_dir_path.exists()
        assert not label_dir_path.exists()
        assert not task2_dir.exists()

    def test_training_data_slicer_skip_border_slices_setting(
        self, rand_int_volume, rand_label_volume, training_settings
    ):
        """Slicer accepts skip_border_slices=True without error (setting stored/logged)."""
        training_settings.use_2_5d_slicing = True
        training_settings.num_slices = 3
        training_settings.skip_border_slices = True
        slicer = TrainingDataSlicer(
            rand_int_volume, rand_label_volume, training_settings
        )
        assert getattr(slicer, "skip_border_slices", False) is True
