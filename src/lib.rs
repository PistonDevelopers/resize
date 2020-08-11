//! Simple resampling library in pure Rust.
//!
//! # Examples
//!
//! ```
//! use resize::Pixel::RGB24;
//! use resize::Type::Lanczos3;
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
pub use px::*;

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
    pub fn new_cubic(b: f32, c: f32) -> Self {
        Self::new(Box::new(move |x| cubic_bc(b, c, x)), 2.0)
    }

    /// Helper to create Lanczos filter with custom radius.
    #[must_use]
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
#[inline]
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

#[inline]
fn lanczos(taps: f32, x: f32) -> f32 {
    if x.abs() < taps {
        sinc(x) * sinc(x / taps)
    } else {
        0.0
    }
}

/// Supported pixel formats.
// TODO(Kagami): YUV planes?
#[allow(non_snake_case)]
pub mod Pixel {
    /// Grayscale, 8-bit.
    #[derive(Debug, Clone, Copy)]
    pub struct Gray8;
    /// Grayscale, 16-bit, native endian.
    #[derive(Debug, Clone, Copy)]
    pub struct Gray16;
    /// RGB, 8-bit per component.
    #[derive(Debug, Clone, Copy)]
    pub struct RGB24;
    /// RGB, 16-bit per component, native endian.
    #[derive(Debug, Clone, Copy)]
    pub struct RGB48;
    /// RGBA, 8-bit per component.
    #[derive(Debug, Clone, Copy)]
    pub struct RGBA;
    /// RGBA, 16-bit per component, native endian.
    #[derive(Debug, Clone, Copy)]
    pub struct RGBA64;
}

/// Resampler with preallocated buffers and coeffecients for the given
/// dimensions and filter type.
#[derive(Debug)]
pub struct Resizer<Format: PixelFormat> {
    // Source/target dimensions.
    w1: usize,
    h1: usize,
    w2: usize,
    h2: usize,
    pix_fmt: Format,
    // Temporary/preallocated stuff.
    tmp: Vec<f32>,
    coeffs_w: Vec<CoeffsLine>,
    coeffs_h: Vec<CoeffsLine>,
}

#[derive(Debug, Clone)]
struct CoeffsLine {
    start: usize,
    coeffs: Arc<[f32]>,
}

impl<Format: PixelFormat> Resizer<Format> {
    /// Create a new resizer instance.
    pub fn new(source_width: usize, source_heigth: usize, dest_width: usize, dest_height: usize, pixel_format: Format, filter_type: Type) -> Self {
        let filter = match filter_type {
            Type::Point => Filter::new(Box::new(point_kernel), 0.0),
            Type::Triangle => Filter::new(Box::new(triangle_kernel), 1.0),
            Type::Catrom => Filter::new_cubic(0.0, 0.5),
            Type::Mitchell => Filter::new_cubic(1.0/3.0, 1.0/3.0),
            Type::Lanczos3 => Filter::new_lanczos(3.0),
            Type::Custom(f) => f,
        };
        // filters very often create repeating patterns,
        // so overall memory used by them can be reduced
        // which should save some cache space
        let mut recycled_coeffs = HashMap::new();

        let coeffs_w = Self::calc_coeffs(source_width, dest_width, &filter, &mut recycled_coeffs);
        let coeffs_h = if source_heigth == source_width && dest_height == dest_width {
            coeffs_w.clone()
        } else {
            Self::calc_coeffs(source_heigth, dest_height, &filter, &mut recycled_coeffs)
        };
        Self {
            w1: source_width,
            h1: source_heigth,
            w2: dest_width,
            h2: dest_height,
            tmp: Vec::with_capacity(source_width * dest_height * pixel_format.get_ncomponents()),
            pix_fmt: pixel_format,
            coeffs_w,
            coeffs_h,
        }
    }

    fn calc_coeffs(s1: usize, s2: usize, f: &Filter, recycled_coeffs: &mut HashMap<(usize, u64), Arc<[f32]>>) -> Vec<CoeffsLine> {
        let ratio = s1 as f32 / s2 as f32;
        // Scale the filter when downsampling.
        let filter_scale = ratio.max(1.);
        let filter_radius = (f.support * filter_scale).ceil();
        (0..s2).map(|x2| {
            let x1 = (x2 as f32 + 0.5) * ratio - 0.5;
            let start = (x1 - filter_radius).ceil() as isize;
            let start = Self::clamp(start, 0, s1 as isize - 1) as usize;
            let end = (x1 + filter_radius).floor() as isize;
            let end = Self::clamp(end, 0, s1 as isize - 1) as usize;
            let sum: f32 = (start..=end).map(|i| (f.kernel)((i as f32 - x1) / filter_scale)).sum();
            let key = (end - start, ((x1 - start as f32) * 100_000.) as u64);
            let coeffs = recycled_coeffs.entry(key).or_insert_with(|| {
                (start..=end).map(|i| {
                    let v = (f.kernel)((i as f32 - x1) / filter_scale);
                    v / sum
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

    // Resample W1xH1 to W1xH2.
    // Stride is a length of the source row (>= W1)
    fn sample_rows(&mut self, src: &[Format::Subpixel], stride: usize) {
        let ncomp = self.pix_fmt.get_ncomponents();
        self.tmp.clear();
        assert!(self.tmp.capacity() <= self.w1 * self.h2 * ncomp); // no reallocations
        for x1 in 0..self.w1 {
            let h2 = self.h2;
            let coeffs_h = &self.coeffs_h[0..h2];
            for y2 in 0..h2 {
                let mut accum = Format::new_accum();
                let line = &coeffs_h[y2];
                let src = &src[(line.start * stride + x1) * ncomp..];
                for (i, coeff) in line.coeffs.iter().copied().enumerate() {
                    let base = (i * stride) * ncomp;
                    let src = &src[base..base + ncomp];
                    for (acc, s) in accum.as_mut().iter_mut().zip(src) {
                        *acc += Format::from_subpixel(s) * coeff;
                    }
                }
                for &v in accum.as_ref().iter() {
                    self.tmp.push(v);
                }
            }
        }
    }

    // Resample W1xH2 to W2xH2.
    fn sample_cols(&mut self, dst: &mut [Format::Subpixel]) {
        let ncomp = self.pix_fmt.get_ncomponents();
        let mut offset = 0;
        // Assert that dst is large enough
        let dst = &mut dst[0..self.h2 * self.w2 * ncomp];
        for y2 in 0..self.h2 {
            let w2 = self.w2;
            let coeffs_w = &self.coeffs_w[0..w2];
            for x2 in 0..w2 {
                let mut accum = Format::new_accum();
                let line = &coeffs_w[x2];
                for (i, coeff) in line.coeffs.iter().copied().enumerate() {
                    let x0 = line.start + i;
                    let base = (x0 * self.h2 + y2) * ncomp;
                    let tmp = &self.tmp[base..base + ncomp];
                    for (acc, &p) in accum.as_mut().iter_mut().zip(tmp) {
                        *acc += p * coeff;
                    }
                }
                for &v in accum.as_ref().iter() {
                    dst[offset] = Format::into_subpixel(v);
                    offset += 1;
                }
            }
        }
    }

    /// Resize `src` image data into `dst`.
    pub fn resize(&mut self, src: &[Format::Subpixel], dst: &mut [Format::Subpixel]) {
        let stride = self.w1;
        self.resize_stride(src, stride, dst)
    }

    /// Resize `src` image data into `dst`, skipping `stride` pixels each row.
    pub fn resize_stride(&mut self, src: &[Format::Subpixel], src_stride: usize, dst: &mut [Format::Subpixel]) {
        // TODO(Kagami):
        // * Multi-thread
        // * Bound checkings
        // * SIMD
        assert!(self.w1 <= src_stride);
        assert!(src.len() >= src_stride * self.h1 * self.pix_fmt.get_ncomponents());
        assert_eq!(dst.len(), self.w2 * self.h2 * self.pix_fmt.get_ncomponents());
        self.sample_rows(src, src_stride);
        self.sample_cols(dst)
    }
}

/// Create a new resizer instance. Alias for `Resizer::new`.
pub fn new<Format: PixelFormat>(src_width: usize, src_height: usize, dest_width: usize, dest_height: usize, pixel_format: Format, filter_type: Type) -> Resizer<Format> {
    Resizer::new(src_width, src_height, dest_width, dest_height, pixel_format, filter_type)
}

/// Resize image data to the new dimension in a single step.
///
/// **NOTE:** If you need to resize to the same dimension multiple times,
/// consider creating an resizer instance since it's faster.
pub fn resize<Format: PixelFormat>(
    src_width: usize, src_height: usize, dest_width: usize, dest_height: usize,
    pixel_format: Format, filter_type: Type,
    src: &[Format::Subpixel], dst: &mut [Format::Subpixel],
) {
    Resizer::new(src_width, src_height, dest_width, dest_height, pixel_format, filter_type).resize(src, dst)
}

#[test]
fn pixel_sizes() {
    assert_eq!(Pixel::RGB24.get_ncomponents(), 3);
    assert_eq!(Pixel::RGB24.get_size(), 3 * 1);
    assert_eq!(Pixel::RGBA.get_size(), 4 * 1);

    assert_eq!(Pixel::RGB48.get_ncomponents(), 3);
    assert_eq!(Pixel::RGB48.get_size(), 3 * 2);
    assert_eq!(Pixel::RGBA64.get_ncomponents(), 4);
    assert_eq!(Pixel::RGBA64.get_size(), 4 * 2);
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
