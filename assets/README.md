# Assets

## face_detection_yunet_2023mar.onnx

YuNet face detection model (Apache-2.0) from OpenCV Zoo:
https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet

- 232 589 bytes, SHA-256 starts with `8f2383e4dd3cfbb4`
- Input: `input` — 1×320×320×3 float32, BGR, 0..255 (no normalization in-graph)
- Outputs: 13 raw heads (`cls_{8,16,32}`, `obj_{8,16,32}`, `bbox_{8,16,32}`,
  `kps_{8,16,32}`) — the decode + NMS postprocess lives in `src/track/`.

Embedded via `include_bytes!` so the binary has no runtime asset dependency.
