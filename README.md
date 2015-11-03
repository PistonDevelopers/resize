# resize [![Build Status](https://travis-ci.org/PistonDevelopers/resize.png?branch=master)](https://travis-ci.org/PistonDevelopers/resize) [![crates.io](https://img.shields.io/crates/v/resize.svg)](https://crates.io/crates/resize)

Simple resampling library in pure Rust.

Features:
* No dependencies, minimal abstractions
* No encoders/decoders, meant to be used with some external library
* Tuned for resizing to the same dimensions multiple times: uses preallocated buffers and matrixes
* Tuned to have result as close as possible to ImageMagick

## Usage

```rust
extern crate resize;
use resize::Type::Triangle;
let mut src = vec![0;w1*h1];
let mut dst = vec![0;w2*h2];
let mut resizer = resize::new(w1, h1, w2, h2, Triangle);
resizer.run(&src, &mut dst);
```

See [API documentation](http://docs.piston.rs/resize/resize/) for overview of all available methods. See also [this example](examples/resize.rs).

## Triangle test

Comparision of IM with libswscale:

```bash
cd examples
convert tiger.png -filter Triangle -resize 540x360 im.png
ffmpeg -i tiger.png -vf scale=540:360:flags=bilinear sws.png
compare sws.png im.png -compose src diff-sws-im.png
```

![](https://raw.githubusercontent.com/PistonDevelopers/resize/master/examples/diff-sws-im.png)

Comparision of this library with IM:

```bash
../target/debug/examples/resize tiger.png 540x360 rust.png
compare rust.png im.png -compose src diff-rust-im.png
```

![](https://raw.githubusercontent.com/PistonDevelopers/resize/master/examples/diff-rust-im.png)

## License

* Library licensed under [MIT](LICENSE)
* Image used in examples licensed under [CC BY-SA 3.0](https://commons.wikimedia.org/wiki/File%3A08-2011._Panthera_tigris_tigris_-_Texas_Park_-_Lanzarote_-TP04.jpg)
