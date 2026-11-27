# A layer keeps its alpha whatever its first frame

## Defect

`podagent/mograph.py` `_pack` encoded each layer's PNG sequence with
`ffmpeg -framerate 30 -pattern_type glob -i <seqdir>/*.png -c:v qtrle …` and no `-pix_fmt`.
The renderer writes a fully opaque PNG frame as RGB and a frame with transparency as RGBA.
ffmpeg takes the stream format from the first frame, so a layer whose first frame is opaque
is encoded as `rgb24`, and every later transparent pixel is packed as opaque black. The
overlay then covers the base for the layer's whole window (seen as the speaker vanishing
behind a card for several seconds).

## Encoder line

`ffmpeg -h encoder=qtrle` (local, 2026-11-17):

    Supported pixel formats: rgb24 rgb555be argb gray

`argb` is the qtrle alpha format.

## Reproduction

One opaque RGB PNG as frame 1, one RGBA PNG transparent at its corners as frame 2, packed
with qtrle and decoded (`-pix_fmt rgba -f rawvideo`):

- without `-pix_fmt argb`: corner pixel of frame 2 is black, opaque (alpha 255)
- with `-pix_fmt argb`: corner pixel of frame 2 has alpha 0

## Fix

`_QTRLE_PIX_FMT = ["-pix_fmt", "argb"]` is added to the qtrle command. No other ffmpeg setting
changes; the BT.709 tags stay. Guarded by
`tests/test_mograph.py::test_a_layer_keeps_its_alpha_after_an_opaque_first_frame`, which runs
real ffmpeg.
