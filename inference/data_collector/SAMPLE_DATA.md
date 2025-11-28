# Sample Data Shape

```json
{
  "session_id": "Vincent732_2025-11-28_19-33-10",
  "participant_name": "Vincent732",
  "start_time": "2025-11-28T19:33:10.259721",
  "metadata": {
    "system_info": {
      "os": "Darwin 25.1.0 (arm64)",
      "python_version": "3.12.9",
      "torch_version": "2.8.0",
      "cpu": {
        "brand": "Apple M3",
        "arch": "arm64",
        "base_speed_ghz": 0.0,
        "flags": [],
        "l2_cache_size_kb": 0,
        "l3_cache_size_kb": 0,
        "physical_cores": 8,
        "total_cores": 8,
        "current_usage_percent": 0.0
      },
      "ram": {
        "total_gb": 16.0,
        "available_gb": 5.95,
        "usage_percent": 62.8
      }
    },
    "screen_size": {
      "width": 1512,
      "height": 982
    },
    "camera_resolution": {
      "width": 1280,
      "height": 720
    },
    "performance": {
      "inference_fps": 53.44
    },
    "video_files": {
      "webcam": "session_Vincent732_2025-11-28_19-33-10.mp4",
      "screen": "session_Vincent732_2025-11-28_19-33-10_screen.mp4"
    }
  },
  "calibration": {
    "baseline_head_pose": {
      "roll": 1.7276810493053416,
      "eye_distance": 0.14060405660672848,
      "eye_center_x": 0.5102538350767275,
      "eye_center_y": 0.491015100645465,
      "num_samples": 9
    },
    "calibration_points": [
      {
        "target": {
          "x": 60,
          "y": 78
        },
        "click": {
          "x": 67,
          "y": 85
        },
        "gaze_result": {
          "gaze": {
            "pitch": 19.773101806640625,
            "yaw": 3.5293731689453125
          },
          "eye_center_x": 0.5074572203504882,
          "eye_center_y": 0.4820538947782015,
          "eye_distance": 0.13943612981499567,
          "roll": 3.0000928553721344
        },
        "timestamp": "2025-11-28T19:33:12.531609",
        "video_timestamp": 0.912598
      },
      // other samples 7 ...
      {
        "target": {
          "x": 1451,
          "y": 903
        },
        "click": {
          "x": 1452,
          "y": 904
        },
        "gaze_result": {
          "gaze": {
            "pitch": -17.791900634765625,
            "yaw": -18.48785400390625
          },
          "eye_center_x": 0.5066168078521536,
          "eye_center_y": 0.5067045029640348,
          "eye_distance": 0.13925419023970564,
          "roll": 0.6567266127852078
        },
        "timestamp": "2025-11-28T19:33:38.505061",
        "video_timestamp": 26.886052
      }
    ]
  },
  "click_events": {
    "explicit_points": [
      {
        "target": {
          "x": 148,
          "y": 149
        },
        "click": {
          "x": 151,
          "y": 146
        },
        "gaze_result": {
          "gaze": {
            "pitch": 17.038467407226562,
            "yaw": 0.3923187255859375
          },
          "eye_center_x": 0.5068490657486484,
          "eye_center_y": 0.4888462402698085,
          "eye_distance": 0.1405316941969511,
          "roll": 1.9321219149030533
        },
        "timestamp": "2025-11-28T19:33:58.420074",
        "video_timestamp": 46.801071
      },
      {
        "target": {
          "x": 600,
          "y": 474
        },
        "click": {
          "x": 597,
          "y": 475
        },
        "gaze_result": {
          "gaze": {
            "pitch": 5.33306884765625,
            "yaw": -7.60821533203125
          },
          "eye_center_x": 0.5041005446104654,
          "eye_center_y": 0.49215513713785863,
          "eye_distance": 0.142815320006439,
          "roll": 2.4364750116491214
        },
        "timestamp": "2025-11-28T19:34:15.129391",
        "video_timestamp": 63.510387
      },
      // other samples ...
      {
        "target": {
          "x": 1000,
          "y": 632
        },
        "click": {
          "x": 1001,
          "y": 641
        },
        "gaze_result": {
          "gaze": {
            "pitch": -5.3500823974609375,
            "yaw": -12.399581909179688
          },
          "eye_center_x": 0.5127356260056499,
          "eye_center_y": 0.5546083304708792,
          "eye_distance": 0.1412494587138157,
          "roll": 3.848344545179344
        },
        "timestamp": "2025-11-28T19:55:30.974807",
        "video_timestamp": 1339.355802
      }
    ],
    "implicit_clicks": [
      {
        "click": {
          "x": 942,
          "y": 951
        },
        "gaze_result": {
          "gaze": {
            "pitch": -5.1436614990234375,
            "yaw": -17.070907592773438
          },
          "eye_center_x": 0.5078757087555505,
          "eye_center_y": 0.4992959952190397,
          "eye_distance": 0.13920414438081374,
          "roll": 0.6778054636267113
        },
        "timestamp": "2025-11-28T19:33:50.741451",
        "video_timestamp": 39.122445
      },
      {
        "click": {
          "x": 860,
          "y": 636
        },
        "gaze_result": {
          "gaze": {
            "pitch": -2.83489990234375,
            "yaw": -8.359588623046875
          },
          "eye_center_x": 0.5098085375070583,
          "eye_center_y": 0.4960656904905079,
          "eye_distance": 0.141063901122348,
          "roll": 1.2218014597200084
        },
        "timestamp": "2025-11-28T19:33:51.831091",
        "video_timestamp": 40.212087
      },
      // other samples ...
      {
        "click": {
          "x": 1031,
          "y": 71
        },
        "gaze_result": {
          "gaze": {
            "pitch": -7.968719482421875,
            "yaw": 5.5198974609375
          },
          "eye_center_x": 0.5085239551775269,
          "eye_center_y": 0.5484296393250216,
          "eye_distance": 0.14266531371417063,
          "roll": 1.6038265161821268
        },
        "timestamp": "2025-11-28T19:55:33.760534",
        "video_timestamp": 1342.141528
      }
    ]
  },
  "video_file": "session_Vincent732_2025-11-28_19-33-10.mp4",
  "end_time": "2025-11-28T19:55:34.347182",
  "statistics": {
    "total_clicks": 1101,
    "calibration_points": 9,
    "explicit_points": 79,
    "implicit_clicks": 1013
  }
}
```
