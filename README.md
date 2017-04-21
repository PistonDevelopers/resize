# resize [![Build Status](https://travis-ci.org/PistonDevelopers/resize.png?branch=master)](https://travis-ci.org/PistonDevelopers/resize) [![crates.io](https://img.shields.io/crates/v/resize.svg)](https://crates.io/crates/resize)

Simple resampling library in pure Rust.

## Features

* No dependencies, minimal abstractions
* No encoders/decoders, meant to be used with some external library
* Tuned for resizing to the same dimensions multiple times: uses preallocated buffers and matrixes

## Usage

```rust
extern crate resize;
use resize::Pixel::RGB24;
use resize::Type::Lanczos3;

// Downscale by 2x.
let (w1, h1) = (640, 480);
let (w2, h2) = (320, 240);
// Don't forget to fill `src` with image data (RGB24).
let src = vec![0;w1*h1*3];
// Destination buffer. Must be mutable.
let mut dst = vec![0;w2*h2*3];
// Create reusable instance.
let mut resizer = resize::new(w1, h1, w2, h2, RGB24, Lanczos3);
// Do resize without heap allocations.
// Might be executed multiple times for different `src` or `dst`.
resizer.resize(&src, &mut dst);
```

See [API documentation](http://docs.piston.rs/resize/resize/) for overview of all available methods. See also [this example](examples/resize.rs).

## Recommendations

Read [this](http://www.imagemagick.org/Usage/filter/) and [this](http://www.imagemagick.org/Usage/filter/nicolas/) great articles on image resizing technics and resampling filters. Tldr; (with built-in filters of this library) use `Lanczos3` for downscaling, use `Mitchell` for upscaling. You may also want to [downscale in linear colorspace](http://www.imagemagick.org/Usage/resize/#resize_colorspace) (but not upscale). Gamma correction routines currently not included to the library, but actually quite simple to accomplish manually, see [here](https://en.wikipedia.org/wiki/Gamma_correction) for some basic theory.

## License

* Library is licensed under [MIT](LICENSE)
* Image used in examples is licensed under [CC BY-SA 3.0](https://commons.wikimedia.org/wiki/File%3A08-2011._Panthera_tigris_tigris_-_Texas_Park_-_Lanzarote_-TP04.jpg)
