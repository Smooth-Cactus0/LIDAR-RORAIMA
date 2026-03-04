import pandas as pd

from lidar_roraima.validation import validate_feature_schema, validate_manifest_schema


def test_manifest_schema_ok():
    df = pd.DataFrame(
        [
            {
                "tile_id": "NP_T-0001",
                "file_name": "NP_T-0001.laz",
                "is_copc": False,
                "is_duplicate": False,
                "las_version": "1.3",
                "point_format": 1,
                "epsg": 31980,
                "bbox": '{"min_x":0,"max_x":1,"min_y":0,"max_y":1,"min_z":0,"max_z":1}',
                "point_count": 1,
                "size_bytes": 1,
                "qa_flags": "[]",
            }
        ]
    )
    report = validate_manifest_schema(df)
    assert report.passed


def test_feature_schema_ok():
    df = pd.DataFrame(
        [
            {
                "tile_id": "NP_T-0001",
                "zone_epsg": 31980,
                "grid_x": 1,
                "grid_y": 2,
                "point_density": 25.0,
                "z_mean": 8.2,
            }
        ]
    )
    report = validate_feature_schema(df)
    assert report.passed
