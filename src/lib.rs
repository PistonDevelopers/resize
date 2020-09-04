//! Simple resampling library in pure Rust.
//!
//! # Examples
//!
//! ```
//! use resize::Pixel::RGB24;
//! use resize::Type::Lanczos3;
//! use rgb::RGB8;
//!
//! // Downscale by 2x.
//! let (w1, h1) = (640, 480);
//! let (w2, h2) = (320, 240);
//! // Don't forget to fill `src` with image data (RGB24).
//! let src = vec![0;w1*h1*3];
//! // Destination buffer. Must be mutable.
//! let mut dst = vec![0;w2*h2*3];
//! // Create reusable instance.
//! let mut resizer = resize::new(w1, h1, w2, h2, RGB24, Lanczos3);
//! // Do resize without heap allocations.
//! // Might be executed multiple times for different `src` or `dst`.
//! resizer.resize(&src, &mut dst);
//! ```
// Current implementation is based on:
// * https://github.com/sekrit-twc/zimg/tree/master/src/zimg/resize
// * https://github.com/PistonDevelopers/image/blob/master/src/imageops/sample.rs
#![deny(missing_docs)]

use std::sync::Arc;
use std::collections::HashMap;
use std::f32;

mod px;
#[allow(deprecated)]
use px::PixelFormatBackCompatShim;
pub use px::PixelFormat;

/// Resizing type to use.
pub enum Type {
    /// Point resizing.
    Point,
    /// Triangle (bilinear) resizing.
    Triangle,
    /// Catmull-Rom (bicubic) resizing.
    Catrom,
    /// Resize using Mitchell-Netravali filter.
    Mitchell,
    /// Resize using Sinc-windowed Sinc with radius of 3.
    Lanczos3,
    /// Resize with custom filter.
    Custom(Filter),
}

/// Resampling filter.
pub struct Filter {
    kernel: Box<dyn Fn(f32) -> f32>,
    support: f32,
}

impl Filter {
    /// Create a new filter.
    ///
    /// # Examples
    ///
    /// ```
    /// use resize::Filter;
    /// fn kernel(x: f32) -> f32 { f32::max(1.0 - x.abs(), 0.0) }
    /// let filter = Filter::new(Box::new(kernel), 1.0);
    /// ```
    #[must_use]
    pub fn new(kernel: Box<dyn Fn(f32) -> f32>, support: f32) -> Self {
        Self { kernel, support }
    }

    /// Helper to create Cubic filter with custom B and C parameters.
    #[must_use]
    #[deprecated(note = "use Type enum")]
    pub fn new_cubic(b: f32, c: f32) -> Self {
        Self::new(Box::new(move |x| cubic_bc(b, c, x)), 2.0)
    }

    /// Helper to create Lanczos filter with custom radius.
    #[must_use]
    #[deprecated(note = "use Type enum")]
    pub fn new_lanczos(radius: f32) -> Self {
        Self::new(Box::new(move |x| lanczos(radius, x)), radius)
    }
}

#[inline]
fn point_kernel(_: f32) -> f32 {
    1.0
}

#[inline]
fn triangle_kernel(x: f32) -> f32 {
    f32::max(1.0 - x.abs(), 0.0)
}

// Taken from
// https://github.com/PistonDevelopers/image/blob/2921cd7/src/imageops/sample.rs#L68
// TODO(Kagami): Could be optimized for known B and C, see e.g.
// https://github.com/sekrit-twc/zimg/blob/1a606c0/src/zimg/resize/filter.cpp#L149
#[inline(always)]
fn cubic_bc(b: f32, c: f32, x: f32) -> f32 {
    let a = x.abs();
    let k = if a < 1.0 {
        (12.0 - 9.0 * b - 6.0 * c) * a.powi(3) +
        (-18.0 + 12.0 * b + 6.0 * c) * a.powi(2) +
        (6.0 - 2.0 * b)
    } else if a < 2.0 {
        (-b - 6.0 * c) * a.powi(3) +
        (6.0 * b + 30.0 * c) * a.powi(2) +
        (-12.0 * b - 48.0 * c) * a +
        (8.0 * b + 24.0 * c)
    } else {
        0.0
    };
    k / 6.0
}

#[inline]
fn sinc(x: f32) -> f32 {
    if x == 0.0 {
        1.0
    } else {
        let a = x * f32::consts::PI;
        a.sin() / a
    }
}

#[inline(always)]
fn lanczos(taps: f32, x: f32) -> f32 {
    if x.abs() < taps {
        sinc(x) * sinc(x / taps)
    } else {
        0.0
    }
}

/// Supported pixel formats.
#[allow(non_snake_case)]
#[allow(non_upper_case_globals)]
pub mod Pixel {
    use std::marker::PhantomData;

    /// shh
    pub(crate) mod generic {
        use std::marker::PhantomData;
        /// RGB pixels
        #[derive(Debug, Copy, Clone)]
        pub struct RgbFormats<InputSubpixel, OutputSubpixel>(pub PhantomData<(InputSubpixel, OutputSubpixel)>);
        /// RGBA pixels
        #[derive(Debug, Copy, Clone)]
        pub struct RgbaFormats<InputSubpixel, OutputSubpixel>(pub PhantomData<(InputSubpixel, OutputSubpixel)>);
        /// Grayscale pixels
        #[derive(Debug, Copy, Clone)]
        pub struct GrayFormats<InputSubpixel, OutputSubpixel>(pub PhantomData<(InputSubpixel, OutputSubpixel)>);
    }
    use self::generic::*;

    /// Grayscale, 8-bit.
    pub const Gray8: GrayFormats<u8, u8> = GrayFormats(PhantomData);
    /// Grayscale, 16-bit, native endian.
    pub const Gray16: GrayFormats<u16, u16> = GrayFormats(PhantomData);

    /// RGB, 8-bit per component.
    pub const RGB24: RgbFormats<u8, u8> = RgbFormats(PhantomData);
    /// RGB, 16-bit per component, native endian.
    pub const RGB48: RgbFormats<u16, u16> = RgbFormats(PhantomData);
    /// RGBA, 8-bit per component.
    pub const RGBA: RgbaFormats<u8, u8> = RgbaFormats(PhantomData);
    /// RGBA, 16-bit per component, native endian.
    pub const RGBA64: RgbaFormats<u16, u16> = RgbaFormats(PhantomData);
}


/// Resampler with preallocated buffers and coeffecients for the given
/// dimensions and filter type.
#[derive(Debug)]
pub struct Resizer<Format: PixelFormat> {
    scale: Scale,
    pix_fmt: Format,
    // Temporary/preallocated stuff.
    tmp: Vec<Format::Accumulator>,
}

#[derive(Debug)]
struct Scale {
    /// Source dimensions.
    w1: usize,
    h1: usize,
    /// Vec's len == target dimensions
    coeffs_w: Vec<CoeffsLine>,
    coeffs_h: Vec<CoeffsLine>,
}

impl Scale {
    #[inline]
    fn w2(&self) -> usize {
        self.coeffs_w.len()
    }

    #[inline]
    fn h2(&self) -> usize {
        self.coeffs_h.len()
    }
}

#[derive(Debug, Clone)]
struct CoeffsLine {
    start: usize,
    coeffs: Arc<[f32]>,
}

type DynCallback<'a> = &'a dyn Fn(f32) -> f32;

impl Scale {
    pub fn new(source_width: usize, source_heigth: usize, dest_width: usize, dest_height: usize, filter_type: Type) -> Self {
        let filter = match filter_type {
            Type::Point => (&point_kernel as DynCallback, 0.0_f32),
            Type::Triangle => (&triangle_kernel as DynCallback, 1.0),
            Type::Catrom => ((&|x| cubic_bc(0.0, 0.5, x)) as DynCallback, 2.0),
            Type::Mitchell => ((&|x| cubic_bc(1.0/3.0, 1.0/3.0, x)) as DynCallback, 2.0),
            Type::Lanczos3 => ((&|x| lanczos(3.0, x)) as DynCallback, 3.0),
            Type::Custom(ref f) => (&f.kernel as DynCallback, f.support),
        };

        // filters very often create repeating patterns,
        // so overall memory used by them can be reduced
        // which should save some cache space
        let mut recycled_coeffs = HashMap::new();

        let coeffs_w = Self::calc_coeffs(source_width, dest_width, filter, &mut recycled_coeffs);
        let coeffs_h = if source_heigth == source_width && dest_height == dest_width {
            coeffs_w.clone()
        } else {
            Self::calc_coeffs(source_heigth, dest_height, filter, &mut recycled_coeffs)
        };

        Self {
            w1: source_width,
            h1: source_heigth,
            coeffs_w,
            coeffs_h,
        }
    }

    fn calc_coeffs(s1: usize, s2: usize, (kernel, support): (&dyn Fn(f32) -> f32, f32), recycled_coeffs: &mut HashMap<(usize, [u8; 4], [u8; 4]), Arc<[f32]>>) -> Vec<CoeffsLine> {
        let ratio = s1 as f64 / s2 as f64;
        // Scale the filter when downsampling.
        let filter_scale = ratio.max(1.);
        let filter_radius = (support as f64 * filter_scale).ceil();
        (0..s2).map(|x2| {
            let x1 = (x2 as f64 + 0.5) * ratio - 0.5;
            let start = (x1 - filter_radius).ceil() as isize;
            let start = Self::clamp(start, 0, s1 as isize - 1) as usize;
            let end = (x1 + filter_radius).floor() as isize;
            let end = Self::clamp(end, 0, s1 as isize - 1) as usize;
            let sum: f64 = (start..=end).map(|i| (kernel)(((i as f64 - x1) / filter_scale) as f32) as f64).sum();
            let key = (end - start, (filter_scale as f32).to_ne_bytes(), (start as f32 - x1 as f32).to_ne_bytes());
            let coeffs = recycled_coeffs.entry(key).or_insert_with(|| {
                (start..=end).map(|i| {
                    let n = ((i as f64 - x1) / filter_scale) as f32;
                    ((kernel)(n.min(support).max(-support)) as f64 / sum) as f32
                }).collect::<Arc<[_]>>()
            }).clone();
            CoeffsLine { start, coeffs }
        }).collect()
    }

    #[inline]
    fn clamp<N: PartialOrd>(input: N, min: N, max: N) -> N {
        if input > max {
            max
        } else if input < min {
            min
        } else {
            input
        }
    }
}

impl<Format: PixelFormat> Resizer<Format> {
    /// Create a new resizer instance.
    pub fn new(source_width: usize, source_heigth: usize, dest_width: usize, dest_height: usize, pixel_format: Format, filter_type: Type) -> Self {
        Self {
            scale: Scale::new(source_width, source_heigth, dest_width, dest_height, filter_type),
            tmp: Vec::new(),
            pix_fmt: pixel_format,
        }
    }

    /// Stride is a length of the source row (>= W1)
    fn resample_both_axes(&mut self, src: &[Format::InputPixel], stride: usize, mut dst: &mut [Format::OutputPixel]) {
        self.tmp.clear();
        self.tmp.reserve(self.scale.w2() * self.scale.h1);

        // Outer loop resamples W2xH1 to W2xH2
        let mut src_rows = src.chunks(stride);
        for row in &self.scale.coeffs_h {
            let w2 = self.scale.w2();

            // Inner loop resamples W1xH1 to W2xH1,
            // but only as many rows as necessary to write a new line
            // to the output
            while self.tmp.len() < w2 * (row.start + row.coeffs.len()) {
                let row = src_rows.next().unwrap();
                let pix_fmt = &self.pix_fmt;
                self.tmp.extend(self.scale.coeffs_w.iter().map(|col| {
                    let mut accum = Format::new();
                    let in_px = &row[col.start..col.start + col.coeffs.len()];
                    for (coeff, in_px) in col.coeffs.iter().copied().zip(in_px.iter().copied()) {
                        pix_fmt.add(&mut accum, in_px, coeff)
                    }
                    accum
                }));
            }

            let tmp_rows = &self.tmp[w2 * row.start..];
            for (col, dst_px) in dst[0..w2].iter_mut().enumerate() {
                let mut accum = Format::new();
                for (coeff, other_row) in row.coeffs.iter().copied().zip(tmp_rows.chunks_exact(w2)) {
                    Format::add_acc(&mut accum, other_row[col], coeff);
                }
                *dst_px = self.pix_fmt.into_pixel(accum);
            }
            dst = &mut dst[w2..];
        }
    }

    /// Resize `src` image data into `dst`.
    pub(crate) fn resize_internal(&mut self, src: &[Format::InputPixel], src_stride: usize, dst: &mut [Format::OutputPixel]) {
        // TODO(Kagami):
        // * Multi-thread
        // * Bound checkings
        // * SIMD
        assert!(self.scale.w1 <= src_stride);
        assert!(src.len() >= src_stride * self.scale.h1);
        assert_eq!(dst.len(), self.scale.w2() * self.scale.h2());
        self.resample_both_axes(src, src_stride, dst);
    }
}

#[allow(deprecated)]
impl<Format: PixelFormatBackCompatShim> Resizer<Format> {
    /// Resize `src` image data into `dst`.
    pub fn resize(&mut self, src: &[Format::Subpixel], dst: &mut [Format::Subpixel]) {
        self.resize_internal(Format::input(src), self.scale.w1, Format::output(dst))
    }

    /// Resize `src` image data into `dst`, skipping `stride` pixels each row.
    pub fn resize_stride(&mut self, src: &[Format::Subpixel], src_stride: usize, dst: &mut [Format::Subpixel]) {
        self.resize_internal(Format::input(src), src_stride, Format::output(dst))
    }
}

/// Create a new resizer instance. Alias for `Resizer::new`.
pub fn new<Format: PixelFormat>(src_width: usize, src_height: usize, dest_width: usize, dest_height: usize, pixel_format: Format, filter_type: Type) -> Resizer<Format> {
    Resizer::new(src_width, src_height, dest_width, dest_height, pixel_format, filter_type)
}

/// Use `new().resize()` instead.
///
/// Resize image data to the new dimension in a single step.
///
/// **NOTE:** If you need to resize to the same dimension multiple times,
/// consider creating an resizer instance since it's faster.
#[deprecated(note="Use resize::new().resize()")]
#[allow(deprecated)]
pub fn resize<Format: PixelFormatBackCompatShim>(
    src_width: usize, src_height: usize, dest_width: usize, dest_height: usize,
    pixel_format: Format, filter_type: Type,
    src: &[Format::Subpixel], dst: &mut [Format::Subpixel],
) {
    Resizer::<Format>::new(src_width, src_height, dest_width, dest_height, pixel_format, filter_type).resize(src, dst)
}

#[test]
fn resize_stride() {
    let mut r = new(2, 2, 3, 4, Pixel::Gray16, Type::Triangle);
    let mut dst = vec![0; 12];
    r.resize_stride(&[
        65535,65535,1,2,
        65535,65535,3,4,
    ], 4, &mut dst);
    assert_eq!(&dst, &[65535; 12]);
}
