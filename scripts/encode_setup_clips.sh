#!/usr/bin/env bash
# Turn screen recordings of the three setup steps into the clips the guided
# setup wizard plays (components/SetupWizard.tsx, StepClip).
#
# The wizard's steps happen in the browser's own chrome -- the bookmarks bar
# appearing, the chip dropped onto it, the bookmark clicked on ESPN -- and
# Playwright records only the viewport, so these cannot be made the way
# scripts/record_demo.py makes demo.webm. Record them by hand instead:
# Chrome, ~1280x800 window, system dark mode, Cmd+Shift+5 -> Record Selected
# Portion around the browser window. One .mov per step:
#
#   bar    the bar hidden on step 1, Cmd+Shift+B, the bar appears     (~4s)
#   drag   on step 2, drag the chip up to the bar and drop it          (~5s)
#   click  on an ESPN fantasy page, click the bookmark; the popup
#          opens and closes itself; the wizard tab reads Connected     (~8s)
#
# Then:  scripts/encode_setup_clips.sh bar.mov drag.mov click.mov
#
# Each clip is cropped to the part of the screen the step is about, trimmed,
# scaled, and given a 1.5s hold on its last frame so the loop breathes
# before it restarts. The crop/trim numbers below are for a 3024x1898
# (Retina 2x) full-screen capture of a window at the top-left; re-tune them
# for a different recording and re-check with a contact sheet:
#   ffmpeg -i out.webm -vf "fps=2,scale=480:-1,tile=4x4" sheet.png
# Requires ffmpeg (brew install ffmpeg). Output: web/public/setup-*.{webm,jpg}
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=web/public
BAR=${1:?bar.mov}; DRAG=${2:?drag.mov}; CLICK=${3:?click.mov}

enc() { # name, input, start, end, vf
  local name=$1 in=$2 ss=$3 to=$4 vf=$5
  ffmpeg -v error -y -ss "$ss" -to "$to" -i "$in" -an \
    -vf "$vf,fps=30,tpad=stop_mode=clone:stop_duration=1.5" \
    -c:v libvpx-vp9 -crf 34 -b:v 0 -row-mt 1 -deadline good "$OUT/setup-$name.webm"
  ffmpeg -v error -y -ss "$ss" -i "$in" -frames:v 1 -vf "$vf" -q:v 4 "$OUT/setup-$name.jpg"
  echo "setup-$name: $(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 "$OUT/setup-$name.webm") $(stat -f %z "$OUT/setup-$name.webm") bytes"
}

# Keep the aspect ratios in step with the `ratio` props in SetupWizard.tsx.
enc bar   "$BAR"   0.4 3.2 "crop=1600:560:0:0,scale=960:-2"
enc drag  "$DRAG"  0.4 4.0 "crop=1700:1100:0:0,scale=850:-2"
enc click "$CLICK" 1.8 7.5 "crop=2000:1200:0:0,scale=960:-2"
