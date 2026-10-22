A print studio finishes every delivered image with the same fixed chain of operations, and the software that applied it is lost. Rebuild it from examples.

`/data/pairs` has 40 before/after pairs named `NN_src.png` (untouched) and `NN_out.png` (finished), all 8-bit sRGB. Pairs 00 to 13 are test charts: grey ramps, flat grey fields, colour patches, a hue sweep, single bright points on black and hard edges. Pairs 14 to 39 are synthetic scenes built from smooth noise, shapes and flat blocks, not camera photographs. Image sizes vary between pairs, up to 1536x1024, and the held-out images were generated the same way as the scenes and also vary in size, within that limit.

Every operation in the chain is an ordinary photo-finishing one: exposure, tone, colour, local contrast, vignette, bloom, grain, or a final quantise to 8 bits with an ordered dither. Which of these are present, how many times, in what order, in which colour space, and with what parameters is for you to find out. Some operations look at the image as a whole or at its size rather than at one pixel at a time. The grain and the dither depend only on pixel position and channel, never on the image content or on the order files are processed.

Write an executable `/app/finish`, called as `/app/finish in.png out.png`, that reads a source PNG and writes the finished PNG as 8-bit RGB with no alpha channel and the same width and height as the input. The checker runs it in a separate container as an unprivileged user, with Python 3.13, NumPy 2.4.4, Pillow 12.1.1 and SciPy 1.17.1 and nothing else installed, so anything the program needs besides the script, such as fitted tables, must be readable files under `/app`. Only `/app` is carried over.

It will be run on 10 held-out images. On every one, no channel at any pixel may differ from the reference output by more than 1 level. All ten have to pass.

You have 14400 seconds to complete this task. Do not cheat by using online solutions or hints specific to this task.
