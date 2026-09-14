# README demo preview

`registers_demo.gif` is a looping preview of the original [`registers_demo.mp4`](../registers_demo.mp4). The README links to the original video for full-quality playback. The preview preserves the full clip and its timing, at 960 pixels wide and 12 frames per second.

To regenerate it from the repository root with FFmpeg (the command refuses to overwrite an existing preview):

```bash
ffmpeg -i registers_demo.mp4 \
  -filter_complex '[0:v]fps=12,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=3' \
  -loop 0 -map_metadata -1 -n media/registers_demo.gif
```
