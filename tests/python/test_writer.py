"""The writer, through its binding: a scene written from CARLA states, read
back with t4-devkit -- TIER IV's own reader, at its latest release (v0.8.0)
-- and passed through its sanity checker."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from autoware_carla_scenario_t4 import T4Writer, __version__

#: The map's offset (MGRS and elevation) and the ego's rear axle.
OFFSET = (1000.0, 2000.0, 30.0)
REAR_AXLE = -1.25


def _jpeg() -> bytes:
    """A small JPEG, as a driver policy's camera frame comes."""
    from io import BytesIO  # noqa: PLC0415

    buf = BytesIO()
    try:
        from PIL import Image  # noqa: PLC0415

        Image.new("RGB", (64, 48)).save(buf, format="JPEG")
        return buf.getvalue()
    except ImportError:
        pytest.skip("Pillow is not installed")


def write_scene(root: Path, *, raw_camera: bool = True) -> Path:
    """Five key frames of an ego driving east at 10 m/s (CARLA's frame), a car
    keeping 10 m ahead in the lane to its right, and a pedestrian far away."""
    data_root = root / "CutIn.t4" / "0"
    w = T4Writer(
        data_root,
        scene_name="CutIn",
        map_offset=OFFSET,
        rear_axle_offset_m=REAR_AXLE,
        max_distance_m=100.0,
        workers=2,
    )
    w.add_sensor("LIDAR_CONCAT", "lidar", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
    w.add_sensor(
        "CAM_FRONT",
        "camera",
        (2.5, 0.0, 1.6),
        (0.5, -0.5, 0.5, -0.5),  # the optical frame, looking forward
        camera_intrinsic=[[32.0, 0.0, 32.0], [0.0, 32.0, 24.0], [0.0, 0.0, 1.0]],
    )
    w.register_actor(2, "car", (0.0, 0.0, 0.7), (1.8, 4.0, 1.4), "npc#2")
    w.register_actor(3, "pedestrian", (0.0, 0.0, 0.9), (0.6, 0.6, 1.8))
    ids = np.array([2, 3], dtype=np.uint64)
    for i in range(5):
        x = 1.0 * i
        w.begin_frame_carla(
            1_700_000_000_000_000 + i * 100_000,
            (x, 0.0, 0.0, 0.0, 0.0, 0.0),
            (10.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        )
        states = np.array(
            [
                [x + 10.0, 3.5, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0],
                [500.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        assert w.add_carla_boxes(ids, states) == 1
        # In base_link: two points on the car (10 m ahead of the actor, so
        # 10 - REAR_AXLE ahead of the rear axle, 3.5 m to the right), one on
        # nothing, one not finite.
        ahead = 10.0 - REAR_AXLE
        w.add_lidar(
            "LIDAR_CONCAT",
            np.array(
                [[ahead, -3.5, 0.7, 0.9], [ahead + 0.5, -3.3, 1.0, 0.8]],
                dtype=np.float32,
            ),
        )
        w.add_lidar(
            "LIDAR_CONCAT",
            np.array([[30.0, 0.0, 0.0, 0.1], [np.nan, 0.0, 0.0, 0.0]], dtype=np.float32),
        )
        if raw_camera:
            w.add_camera("CAM_FRONT", np.full((48, 64, 4), 200, dtype=np.uint8), "bgra")
        else:
            w.add_camera_jpeg("CAM_FRONT", _jpeg(), 64, 48)
        w.add_vehicle_state(accel_pedal=50.0, shift_state="FORWARD", speed=10.0)
        w.end_frame()
    summary = w.finish()
    assert summary["samples"] == 5
    assert summary["sample_annotations"] == 5
    return data_root.parent


def _table(root: Path, name: str) -> list[dict[str, Any]]:
    return json.loads((root / "0" / "annotation" / f"{name}.json").read_text())


def test_the_version_is_the_crates() -> None:
    assert __version__.count(".") == 2


def test_carla_states_become_map_frame_poses(tmp_path: Path) -> None:
    root = write_scene(tmp_path)
    ego = _table(root, "ego_pose")[0]
    np.testing.assert_allclose(
        ego["translation"], [REAR_AXLE + OFFSET[0], OFFSET[1], OFFSET[2]], atol=1e-9
    )
    np.testing.assert_allclose(ego["twist"], [10.0, 0, 0, 0, 0, 0], atol=1e-9)
    box = _table(root, "sample_annotation")[0]
    np.testing.assert_allclose(
        box["translation"], [10.0 + OFFSET[0], -3.5 + OFFSET[1], 0.7 + OFFSET[2]], atol=1e-9
    )
    assert box["size"] == [1.8, 4.0, 1.4]
    assert box["num_lidar_pts"] == 2
    # The pedestrian 500 m away is beyond max_distance_m.
    assert [i["instance_name"] for i in _table(root, "instance")] == ["npc#2"]


def test_a_sweep_drops_the_points_that_are_not_finite(tmp_path: Path) -> None:
    root = write_scene(tmp_path)
    cloud = np.fromfile(root / "0" / "data" / "LIDAR_CONCAT" / "00000.pcd.bin", dtype=np.float32)
    assert cloud.reshape(-1, 5).shape == (3, 5)
    assert (cloud.reshape(-1, 5)[:, 4] == -1.0).all()


def test_misuse_is_a_value_error(tmp_path: Path) -> None:
    w = T4Writer(tmp_path / "x", scene_name="x")
    with pytest.raises(ValueError, match="sensor"):
        w.begin_frame(1, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
    w.add_sensor("LIDAR_CONCAT", "lidar", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="not declared"):
        w.begin_frame(1, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
        w.add_lidar("LIDAR_TOP", np.zeros((1, 4), dtype=np.float32))
    with pytest.raises(RuntimeError, match="no sensor data"):
        w.finish()


@pytest.mark.parametrize("raw_camera", [True, False])
def test_t4_devkit_loads_the_dataset(tmp_path: Path, raw_camera: bool) -> None:
    t4_devkit = pytest.importorskip("t4_devkit")
    from t4_devkit.dataclass import LidarPointCloud  # noqa: PLC0415

    root = write_scene(tmp_path, raw_camera=raw_camera)
    dataset = t4_devkit.T4Devkit(str(root))
    assert dataset.version == "0"
    assert len(dataset.sample) == 5

    sample = dataset.sample[0]
    path, boxes, intrinsic = dataset.get_sample_data(sample.data["LIDAR_CONCAT"])
    assert intrinsic is None
    (box,) = boxes
    assert box.semantic_label.name == "car"
    assert box.num_points == 2
    np.testing.assert_allclose(box.position, [10.0 - REAR_AXLE, -3.5, 0.7], atol=1e-6)
    assert LidarPointCloud.from_file(path).points.shape == (4, 3)

    path, boxes, intrinsic = dataset.get_sample_data(sample.data["CAM_FRONT"])
    np.testing.assert_allclose(intrinsic, [[32.0, 0.0, 32.0], [0.0, 32.0, 24.0], [0.0, 0.0, 1.0]])
    assert Path(path).read_bytes()[:2] == b"\xff\xd8"
    # The car is ahead and to the right: in the optical frame, +z and +x.
    (box,) = boxes
    assert box.position[2] > 0.0 and box.position[0] > 0.0


def test_the_dataset_passes_t4_devkits_sanity_check(tmp_path: Path) -> None:
    pytest.importorskip("t4_devkit")
    from t4_devkit.sanity import sanity_check  # noqa: PLC0415

    result = sanity_check(str(write_scene(tmp_path)))
    assert result.is_passed()
    failed = {r.id for r in result.reports if r.status.name == "FAILED"}
    # Warnings only: a simulated scene has no ROS bag, Lanelet2 map or point
    # cloud map here.
    assert failed <= {"STR005", "STR008", "STR009"}
    assert not [r for r in result.reports if r.severity.is_error() and r.status.name == "FAILED"]


def test_arrays_in_any_memory_order_are_read_by_row(tmp_path: Path) -> None:
    """A Fortran-ordered (or transposed) array is read in its logical order,
    not its memory's."""
    points = np.array([[1.0, 2.0, 3.0, 0.5], [4.0, 5.0, 6.0, 0.25]], dtype=np.float32)
    clouds = []
    for name, array in (("c", points), ("f", np.asfortranarray(points)), ("t", points.T.copy().T)):
        w = T4Writer(tmp_path / name, scene_name=name)
        w.add_sensor("LIDAR_CONCAT", "lidar", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
        w.begin_frame(1, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
        w.add_lidar("LIDAR_CONCAT", array)
        w.end_frame()
        w.finish()
        clouds.append(np.fromfile(tmp_path / name / "data" / "LIDAR_CONCAT" / "00000.pcd.bin", dtype=np.float32))
    for cloud in clouds:
        np.testing.assert_array_equal(cloud.reshape(-1, 5)[:, :4], points)
