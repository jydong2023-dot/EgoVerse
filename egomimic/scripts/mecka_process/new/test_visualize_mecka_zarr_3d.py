import numpy as np

from egomimic.scripts.mecka_process.visualize_mecka_zarr_3d import (
    collect_points_for_bounds,
    default_panel_groups,
    quaternion_wxyz_to_matrix,
)


def test_quaternion_wxyz_to_matrix_identity() -> None:
    matrix = quaternion_wxyz_to_matrix(np.array([1.0, 0.0, 0.0, 0.0]))

    np.testing.assert_allclose(matrix, np.eye(3), atol=1e-7)


def test_collect_points_for_bounds_includes_pose_origins_and_keypoints() -> None:
    arrays = {
        "obs_head_pose": np.array([[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]]),
        "left.obs_keypoints": np.ones((1, 63)),
        "right.obs_keypoints": np.full((1, 63), 2.0),
        "left.obs_wrist_pose": np.array([[4.0, 5.0, 6.0, 1.0, 0.0, 0.0, 0.0]]),
        "right.obs_wrist_pose": np.array([[7.0, 8.0, 9.0, 1.0, 0.0, 0.0, 0.0]]),
        "left.obs_ee_pose": np.array([[10.0, 11.0, 12.0, 1.0, 0.0, 0.0, 0.0]]),
        "right.obs_ee_pose": np.array([[13.0, 14.0, 15.0, 1.0, 0.0, 0.0, 0.0]]),
    }

    points = collect_points_for_bounds(arrays)

    assert points.shape[1] == 3
    assert [1.0, 2.0, 3.0] in points.tolist()
    assert [13.0, 14.0, 15.0] in points.tolist()


def test_default_panel_groups_contains_all_requested_views() -> None:
    assert default_panel_groups() == [
        "head_pose",
        "obs_keypoints",
        "obs_wrist_pose",
        "obs_ee_pose",
        "all",
    ]
