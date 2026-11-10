#!/usr/bin/env bash
# Test the V4L2 camera path in Kadr without a physical webcam.
#
# 1) loads the v4l2loopback module (requires root),
# 2) feeds ffmpeg testsrc2 into it,
# 3) then select "Kadr Test Cam" in the app's camera list.
#
# Usage: ./scripts/camera-loopback-test.sh [/dev/video20]
set -euo pipefail
DEV=${1:-/dev/video20}

if [ ! -e "$DEV" ]; then
  echo "loading v4l2loopback -> $DEV"
  sudo modprobe v4l2loopback video_devices="$DEV" card_label="Kadr Test Cam" formats=yuyv
fi

echo "feeding $DEV with testsrc2 (Ctrl-C to stop)"
echo "now start the app and pick 'Kadr Test Cam' in the camera list"
ffmpeg -hide_banner -loglevel warning -re \
  -f lavfi -i "testsrc2=size=640x480:rate=30" \
  -pix_fmt yuyv422 -f v4l2 "$DEV"
