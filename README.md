# autoware-carla-scenario-t4

Writes the runs of [autoware_carla_scenario](https://github.com/autowarefoundation/autoware_carla_scenario)
as [T4 datasets](https://github.com/tier4/t4-devkit), TIER IV's nuScenes-style
format: the ego's poses, the boxes of every vehicle and pedestrian, the
LiDAR sweeps and the camera frames, which t4-devkit (`T4Devkit`, `t4sanity`)
reads. The writer is Rust; `autoware_carla_scenario_t4.T4Writer` is its PyO3
binding, one abi3 wheel per platform for every CPython from 3.10.

autoware_carla_scenario drives it with `t4.enabled=true` (its `t4` extra and
`docs/t4_dataset.md`):

```bash
pip install 'autoware-carla-scenario[t4]'
uv run scenario ... t4.enabled=true   # the sensors are driver.lidars / driver.cameras
```

## The writer

```python
import numpy as np
from autoware_carla_scenario_t4 import T4Writer

w = T4Writer("MyScene.t4/0", scene_name="MyScene")
w.add_sensor("LIDAR_CONCAT", "lidar", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
w.begin_frame(1_700_000_000_000_000, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
w.add_annotation(7, "car", (10.0, 0.0, 0.75), (1.8, 4.0, 1.5), (1.0, 0.0, 0.0, 0.0))
w.add_lidar("LIDAR_CONCAT", np.zeros((100, 4), dtype=np.float32))
w.end_frame()
print(w.finish())
```

- Declare the sensors (`add_sensor`: a channel, `lidar` or `camera`, its mount
  in the ego's frame; a camera's 3x3 intrinsics), then for every key frame
  `begin_frame`, its boxes and sensor data, `end_frame`; `finish` writes the
  tables and `status.json`.
- Poses are right-handed, quaternions `(w, x, y, z)`; a box's size is
  `(width, length, height)`.
- `begin_frame_carla`, `register_actor` and `add_carla_boxes` take CARLA's
  states (its left-handed world frame, degrees) and turn them into the map
  frame themselves: `y` flipped, the map's offset (`map_offset`) added, the
  ego's pose at its rear axle (`rear_axle_offset_m`), road users beyond
  `max_distance_m` of the ego left out.
- `add_lidar` takes `float32 [N, >=3]` (`x, y, z[, intensity]`); the sweeps a
  channel gets in a frame are concatenated (`LIDAR_CONCAT`) and written as
  T4's `(x, y, z, intensity, ring = -1)`. `add_camera` takes a raw frame
  (`uint8 [H, W, C]`, BGRA by default) and encodes it as JPEG;
  `add_camera_jpeg` takes one already encoded.
- Tokens are derived from `token_seed` (the scene name by default): the same
  frames written twice are the same dataset.

### Performance

It runs on every key frame of a simulation, so the caller's thread does as
little as it can: the CARLA states arrive as one array, a sweep is copied once
into a buffer reused from frame to frame, and worker threads (`workers`) write
the sweeps, drop non-finite points, count each box's points (on a grid over
the boxes, so a point is tested against the few boxes around it) and write or
encode the camera frames, with at most `queue` files waiting. A key frame of
two 60,000-point sweeps, a JPEG frame and 50 road users costs the caller about
0.3 ms.

## Development

```bash
cargo test                                   # the writer
uv venv && uv pip install --group test .     # the wheel and the test group
.venv/bin/python -m pytest                   # read back with t4-devkit v0.8.0
```

The Python tests read every dataset they write back with t4-devkit v0.8.0, its
latest release, and run its sanity checker: every check of severity ERROR
passes. t4-devkit supports Python 3.10-3.12; on later Pythons those tests are
skipped.

## Releasing

As in [autoware_carla_scenario_rerun](https://github.com/hakuturu583/autoware_carla_scenario_rerun):
a PR merged to `main` with exactly one `release:patch` / `release:minor` /
`release:major` label bumps the version (`tools/bump_version.py`: Cargo.toml
and Cargo.lock), tags it `X.Y.Z` and publishes the manylinux_2_28 abi3 wheels
(x86_64, aarch64) and the sdist to PyPI (`.github/workflows/release.yml`). A
pushed `X.Y.Z` tag publishes that version. A manual run (`workflow_dispatch`)
is a dry run: it tests, builds and checks the sdist and both wheels, and
uploads nothing. CI builds the same manylinux wheels on every PR.

The first release went out by label too: 0.1.0 was never published, and the
PR that added this paragraph, labelled `release:patch`, released 0.1.1. A
hand-cut tag is still the way to publish a version without a PR.

Publishing uses PyPI Trusted Publishing, no tokens: the project's publisher is

| | |
|---|---|
| PyPI project name | `autoware-carla-scenario-t4` |
| Owner | `hakuturu583` |
| Repository name | `autoware_carla_scenario_t4` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

## License

Apache-2.0
