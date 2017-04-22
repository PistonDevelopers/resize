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

use std::f32;
use std::mem;

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
    kernel: Box<Fn(f32) -> f32>,
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
    pub fn new(kernel: Box<Fn(f32) -> f32>, support: f32) -> Filter {
        Filter {kernel: kernel, support: support}
    }

    /// Helper to create Cubic filter with custom B and C parameters.
    pub fn new_cubic(b: f32, c: f32) -> Filter {
        Filter::new(Box::new(move |x| cubic_bc(b, c, x)), 2.0)
    }

    /// Helper to create Lanczos filter with custom radius.
    pub fn new_lanczos(radius: f32) -> Filter {
        Filter::new(Box::new(move |x| lanczos(radius, x)), radius)
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

/// See `Pixel`
pub trait PixelFormat: Copy {
    /// Array to hold temporary values.
    type Accumulator: AsRef<[f32]> + AsMut<[f32]>;
    /// Type of a Subpixel of each pixel (8 or 16 bits).
    type Subpixel: Copy + Into<f32>;

    /// New empty Accumulator.
    fn new_accum() -> Self::Accumulator;

    /// Convert float to integer value in range appropriate for this pixel format.
    fn into_subpixel(v: f32) -> Self::Subpixel;

    /// Size of one pixel in that format in bytes.
    fn get_size(&self) -> usize {
        self.get_ncomponents() * mem::size_of::<Self::Subpixel>()
    }

    /// Return number of components of that format.
    #[inline(always)]
    fn get_ncomponents(&self) -> usize {
        Self::new_accum().as_ref().len()
    }
}

impl PixelFormat for Pixel::Gray8 {
    type Accumulator = [f32;1];
    type Subpixel = u8;
    fn new_accum() -> Self::Accumulator {[0.0;1]}
    fn into_subpixel(v: f32) -> Self::Subpixel {
        Resizer::<Self>::pack_u8(v)
    }
}

impl PixelFormat for Pixel::Gray16 {
    type Accumulator = [f32;1];
    type Subpixel = u16;
    fn new_accum() -> Self::Accumulator {[0.0;1]}
    fn into_subpixel(v: f32) -> Self::Subpixel {
        Resizer::<Self>::pack_u16(v)
    }
}

impl PixelFormat for Pixel::RGB24 {
    type Accumulator = [f32;3];
    type Subpixel = u8;
    fn new_accum() -> Self::Accumulator {[0.0;3]}
    fn into_subpixel(v: f32) -> Self::Subpixel {
        Resizer::<Self>::pack_u8(v)
    }
}
impl PixelFormat for Pixel::RGBA  {
    type Accumulator = [f32;4];
    type Subpixel = u8;
    fn new_accum() -> Self::Accumulator {[0.0;4]}
    fn into_subpixel(v: f32) -> Self::Subpixel {
        Resizer::<Self>::pack_u8(v)
    }
}
impl PixelFormat for Pixel::RGB48  {
    type Accumulator = [f32;3];
    type Subpixel = u16;
    fn new_accum() -> Self::Accumulator {[0.0;3]}
    fn into_subpixel(v: f32) -> Self::Subpixel {
        Resizer::<Self>::pack_u16(v)
    }
}
impl PixelFormat for Pixel::RGBA64  {
    type Accumulator = [f32;4];
    type Subpixel = u16;
    fn new_accum() -> Self::Accumulator {[0.0;4]}
    fn into_subpixel(v: f32) -> Self::Subpixel {
        Resizer::<Self>::pack_u16(v)
    }
}

/// Resampler with preallocated buffers and coeffecients for the given
/// dimensions and filter type.
#[derive(Debug)]
pub struct Resizer<Pixel: PixelFormat> {
    // Source/target dimensions.
    w1: usize,
    h1: usize,
    w2: usize,
    h2: usize,
    pix_fmt: Pixel,
    // Temporary/preallocated stuff.
    tmp: Vec<f32>,
    coeffs_w: Vec<CoeffsLine>,
    coeffs_h: Vec<CoeffsLine>,
}

#[derive(Debug)]
struct CoeffsLine {
    left: usize,
    data: Vec<f32>,
}

impl<Pixel: PixelFormat> Resizer<Pixel> {
    /// Create a new resizer instance.
    pub fn new(source_width: usize, source_heigth: usize, dest_width: usize, dest_height: usize, pixel_format: Pixel, filter_type: Type) -> Self {
        let filter = match filter_type {
            Type::Point => Filter::new(Box::new(point_kernel), 0.0),
            Type::Triangle => Filter::new(Box::new(triangle_kernel), 1.0),
            Type::Catrom => Filter::new_cubic(0.0, 0.5),
            Type::Mitchell => Filter::new_cubic(1.0/3.0, 1.0/3.0),
            Type::Lanczos3 => Filter::new_lanczos(3.0),
            Type::Custom(f) => f,
        };
        Resizer {
            w1: source_width,
            h1: source_heigth,
            w2: dest_width,
            h2: dest_height,
            pix_fmt: pixel_format,
            tmp: vec![0.0; source_width * dest_height * pixel_format.get_ncomponents()],
            // TODO(Kagami): Use same coeffs if w1 = h1 = w2 = h2?
            coeffs_w: Self::calc_coeffs(source_width, dest_width, &filter),
            coeffs_h: Self::calc_coeffs(source_heigth, dest_height, &filter),
        }
    }

    fn calc_coeffs(s1: usize, s2: usize, f: &Filter) -> Vec<CoeffsLine> {
        let ratio = s1 as f32 / s2 as f32;
        // Scale the filter when downsampling.
        let filter_scale = if ratio > 1.0 { ratio } else { 1.0 };
        let filter_radius = (f.support * filter_scale).ceil();
        let mut coeffs = Vec::with_capacity(s2);
        for x2 in 0..s2 {
            let x1 = (x2 as f32 + 0.5) * ratio - 0.5;
            let left = (x1 - filter_radius).ceil() as isize;
            let left = Self::clamp(left, 0, s1 as isize - 1) as usize;
            let right = (x1 + filter_radius).floor() as isize;
            let right = Self::clamp(right, 0, s1 as isize - 1) as usize;
            let mut data = Vec::with_capacity(right + 1 - left);
            let mut sum = 0.0;
            for i in left..right+1 {
                sum += (f.kernel)((i as f32 - x1) / filter_scale);
            }
            for i in left..right+1 {
                let v = (f.kernel)((i as f32 - x1) / filter_scale);
                data.push(v / sum);
            }
            coeffs.push(CoeffsLine {left: left, data: data});
        }
        coeffs
    }

    #[inline]
    fn clamp<N: PartialOrd>(input: N, min: N, max: N) -> N {
        if input > max {
            return max;
        } else if input < min {
            return min;
        } else {
            return input;
        }
    }

    #[inline]
    fn pack_u8(v: f32) -> u8 {
        if v > 255.0 {
            return 255;
        } else if v < 0.0 {
            return 0;
        } else {
            return v.round() as u8;
        }
    }

    #[inline]
    fn pack_u16(v: f32) -> u16 {
        if v > 65535.0 {
            return 65535;
        } else if v < 0.0 {
            return 0;
        } else {
            return v.round() as u16;
        }
    }

    // Resample W1xH1 to W1xH2.
    // Stride is a length of the source row (>= W1)
    fn sample_rows(&mut self, src: &[Pixel::Subpixel], stride: usize) {
        let ncomp = self.pix_fmt.get_ncomponents();
        let mut offset = 0;
        for x1 in 0..self.w1 {
            for y2 in 0..self.h2 {
                let mut accum = Pixel::new_accum();
                let ref line = self.coeffs_h[y2];
                for (i, &coeff) in line.data.iter().enumerate() {
                    let y0 = line.left + i;
                    let base = (y0 * stride + x1) * ncomp;
                    let src = &src[base .. base + ncomp];
                    for (acc, &s) in accum.as_mut().iter_mut().zip(src) {
                        *acc += s.into() * coeff;
                    }
                }
                for &v in accum.as_ref().iter() {
                    self.tmp[offset] = v;
                    offset += 1;
                }
            }
        }
    }

    // Resample W1xH2 to W2xH2.
    fn sample_cols(&mut self, dst: &mut [Pixel::Subpixel]) {
        let ncomp = self.pix_fmt.get_ncomponents();
        let mut offset = 0;
        for y2 in 0..self.h2 {
            for x2 in 0..self.w2 {
                let mut accum = Pixel::new_accum();
                let ref line = self.coeffs_w[x2];
                for (i, &coeff) in line.data.iter().enumerate() {
                    let x0 = line.left + i;
                    let base = (x0 * self.h2 + y2) * ncomp;
                    let tmp = &self.tmp[base .. base + ncomp];
                    for (acc, &p) in accum.as_mut().iter_mut().zip(tmp) {
                        *acc += p * coeff;
                    }
                }
                for &v in accum.as_ref().iter() {
                    dst[offset] = Pixel::into_subpixel(v);
                    offset += 1;
                }
            }
        }
    }

    /// Resize `src` image data into `dst`.
    pub fn resize(&mut self, src: &[Pixel::Subpixel], dst: &mut [Pixel::Subpixel]) {
        let stride = self.w1;
        self.resize_stride(src, stride, dst)
    }

    /// Resize `src` image data into `dst`, skipping `stride` pixels each row.
    pub fn resize_stride(&mut self, src: &[Pixel::Subpixel], src_stride: usize, dst: &mut [Pixel::Subpixel]) {
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
pub fn new<Pixel: PixelFormat>(src_width: usize, src_height: usize, dest_width: usize, dest_height: usize, pixel_format: Pixel, filter_type: Type) -> Resizer<Pixel> {
    Resizer::new(src_width, src_height, dest_width, dest_height, pixel_format, filter_type)
}

/// Resize image data to the new dimension in a single step.
///
/// **NOTE:** If you need to resize to the same dimension multiple times,
/// consider creating an resizer instance since it's faster.
pub fn resize<Pixel: PixelFormat>(
    src_width: usize, src_height: usize, dest_width: usize, dest_height: usize,
    pixel_format: Pixel, filter_type: Type,
    src: &[Pixel::Subpixel], dst: &mut [Pixel::Subpixel],
) {
    Resizer::new(src_width, src_height, dest_width, dest_height, pixel_format, filter_type).resize(src, dst)
}

#[test]
fn pixel_sizes() {
    assert_eq!(Pixel::RGB24.get_ncomponents(), 3);
    assert_eq!(Pixel::RGB24.get_size(), 3*1);
    assert_eq!(Pixel::RGBA.get_size(), 4*1);

    assert_eq!(Pixel::RGB48.get_ncomponents(), 3);
    assert_eq!(Pixel::RGB48.get_size(), 3*2);
    assert_eq!(Pixel::RGBA64.get_ncomponents(), 4);
    assert_eq!(Pixel::RGBA64.get_size(), 4*2);
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
