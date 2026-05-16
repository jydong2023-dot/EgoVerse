import numpy as np

from egomimic.scripts.mecka_process.zarr_to_lerobot3 import (
    build_lerobot_features,
    prestack_future,
)


def test_prestack_future_repeats_last_frame_when_chunk_exceeds_episode() -> None:
    values = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)

    stacked = prestack_future(values, chunk_size=4)

    assert stacked.shape == (3, 4, 1)
    np.testing.assert_array_equal(stacked[0, :, 0], np.array([1.0, 2.0, 3.0, 3.0]))
    np.testing.assert_array_equal(stacked[2, :, 0], np.array([3.0, 3.0, 3.0, 3.0]))


def test_build_lerobot_features_uses_lerobot_image_and_mecka_action_keys() -> None:
    features = build_lerobot_features(
        image_shape=(360, 640, 3),
        chunk_size=100,
        encode_video=True,
    )

    assert features["observations.images.front_img_1"]["dtype"] == "video"
    assert features["observations.images.front_img_1"]["shape"] == (3, 360, 640)
    assert features["observations.state.ee_pose"]["shape"] == (14,)
    assert features["actions_ee_keypoints_world"]["shape"] == (100, 126)
