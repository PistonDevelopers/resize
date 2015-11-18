//! Simple resampling library in pure Rust.
//!
//! # Examples
//!
//! ```ignore
//! extern crate resize;
//! use resize::Pixel::Gray8;
//! use resize::Type::Triangle;
//! let mut src = vec![0;w1*h1];
//! let mut dst = vec![0;w2*h2];
//! let mut resizer = resize::new(w1, h1, w2, h2, Gray8, Triangle);
//! resizer.resize(&src, &mut dst);
//! ```
// Current implementation is based on:
// * https://github.com/sekrit-twc/zimg/tree/master/src/zimg/resize
// * https://github.com/PistonDevelopers/image/blob/master/src/imageops/sample.rs
#![deny(missing_docs)]

use std::f32;

/// Resizing type to use.
pub enum Type {
    /// Point resizing.
    Point,
    /// Box (nearest) resizing.
    Nearest,
    /// Triangle (bilinear) resizing.
    Triangle,
    /// Catmull-Rom (bicubic) resizing.
    Catrom,
    /// Resize using sinc-windowed filter with radius of 2.
    Lanczos2,
    /// Sinc-windowed filter with radius of 3.
    Lanczos3,
    /// Resize using custom filter.
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
    /// fn kernel(x: f32) -> f32 { f32::round(x) }
    /// let filter = Filter::new(Box::new(kernel), 0.5);
    /// ```
    pub fn new(kernel: Box<Fn(f32) -> f32>, support: f32) -> Filter {
        Filter {kernel: kernel, support: support}
    }
}

#[inline]
fn point_kernel(_: f32) -> f32 {
    1.0
}

#[inline]
fn box_kernel(x: f32) -> f32 {
    if x.abs() <= 0.5 { 1.0 } else { 0.0 }
}

#[inline]
fn triangle_kernel(x: f32) -> f32 {
    f32::max(1.0 - x.abs(), 0.0)
}

// Taken from
// https://github.com/PistonDevelopers/image/blob/2921cd7/src/imageops/sample.rs#L68
// Probably may be optimized a bit, see e.g.
// https://github.com/sekrit-twc/zimg/blob/1a606c0/src/zimg/resize/filter.cpp#L149
#[inline]
fn bc_cubic_spline(b: f32, c: f32, x: f32) -> f32 {
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
fn lanczos_kernel(taps: f32, x: f32) -> f32 {
    if x.abs() < taps {
        sinc(x) * sinc(x / taps)
    } else {
        0.0
    }
}

/// Supported pixel formats.
// TODO(Kagami): >8-bit formats.
#[derive(Debug, Clone, Copy)]
pub enum Pixel {
    /// Grayscale, 8-bit.
    Gray8,
    /// RGB, 8-bit per component.
    RGB24,
    /// RGBA, 8-bit per component.
    RGBA,
}

impl Pixel {
    /// Size of one pixel in that format in bytes.
    #[inline]
    pub fn get_size(&self) -> usize {
        match *self {
            Pixel::Gray8 => 1,
            Pixel::RGB24 => 3,
            Pixel::RGBA => 4,
        }
    }

    /// Return number of components of that format.
    #[inline]
    pub fn get_ncomponents(&self) -> usize {
        match *self {
            Pixel::Gray8 => 1,
            Pixel::RGB24 => 3,
            Pixel::RGBA => 4,
        }
    }
}

/// Resampler with preallocated buffers and coeffecients for the given
/// dimensions and filter type.
#[derive(Debug)]
pub struct Resizer {
    // Source/target dimensions.
    w1: usize,
    h1: usize,
    w2: usize,
    h2: usize,
    pix_fmt: Pixel,
    // Temporary/preallocated stuff.
    tmp: Vec<f32>,
    accum: Vec<f32>,
    coeffs_w: Vec<CoeffsLine>,
    coeffs_h: Vec<CoeffsLine>,
}

#[derive(Debug)]
struct CoeffsLine {
    left: usize,
    data: Vec<f32>,
}

impl Resizer {
    /// Create a new resizer instance.
    pub fn new(w1: usize, h1: usize, w2: usize, h2: usize, p: Pixel, t: Type) -> Resizer {
        let filter = match t {
            Type::Point     => Filter::new(Box::new(point_kernel),                     0.0),
            Type::Nearest   => Filter::new(Box::new(box_kernel),                       0.5),
            Type::Triangle  => Filter::new(Box::new(triangle_kernel),                  1.0),
            Type::Catrom    => Filter::new(Box::new(|x| bc_cubic_spline(0.0, 0.5, x)), 2.0),
            Type::Lanczos2  => Filter::new(Box::new(|x| lanczos_kernel(2.0, x)),       2.0),
            Type::Lanczos3  => Filter::new(Box::new(|x| lanczos_kernel(3.0, x)),       3.0),
            Type::Custom(f) => f,
        };
        Resizer {
            w1: w1,
            h1: h1,
            w2: w2,
            h2: h2,
            pix_fmt: p,
            tmp: vec![0.0;w1*h2*p.get_size()],
            accum: vec![0.0;p.get_ncomponents()],
            // TODO(Kagami): Use same coeffs if w1 = h1 = w2 = h2?
            coeffs_w: Self::calc_coeffs(w1, w2, &filter),
            coeffs_h: Self::calc_coeffs(h1, h2, &filter),
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
            let mut data = Vec::with_capacity(right - left + 1);
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
    fn clamp<N: PartialOrd>(v: N, min: N, max: N) -> N {
        if v <= min {
            min
        } else if v >= max {
            max
        } else {
            v
        }
    }

    #[inline]
    fn pack_u8(mut v: f32) -> u8 {
        v = v.round();
        v = f32::min(f32::max(v, 0.0), 255.0);
        v as u8
    }

    #[inline]
    fn clear_accum(&mut self) {
        for v in self.accum.iter_mut() {
            *v = 0.0;
        }
    }

    // Resample W1xH1 to W1xH2.
    fn sample_rows(&mut self, src: &[u8]) {
        let ncomp = self.pix_fmt.get_ncomponents();
        let mut offset = 0;
        for x1 in 0..self.w1 {
            for y2 in 0..self.h2 {
                self.clear_accum();
                let ref line = self.coeffs_h[y2];
                for (i, coeff) in line.data.iter().enumerate() {
                    let y0 = line.left + i;
                    let base = (y0 * self.w1 + x1) * ncomp;
                    for n in 0..ncomp {
                        let p = src[base + n] as f32;
                        self.accum[n] += p * coeff;
                    }
                }
                for &v in &self.accum {
                    self.tmp[offset] = v;
                    offset += 1;
                }
            }
        }
    }

    // Resample W1xH2 to W2xH2.
    fn sample_cols(&mut self, dst: &mut [u8]) {
        let ncomp = self.pix_fmt.get_ncomponents();
        let mut offset = 0;
        for y2 in 0..self.h2 {
            for x2 in 0..self.w2 {
                self.clear_accum();
                let ref line = self.coeffs_w[x2];
                for (i, coeff) in line.data.iter().enumerate() {
                    let x0 = line.left + i;
                    let base = (x0 * self.h2 + y2) * ncomp;
                    for n in 0..ncomp {
                        let p = self.tmp[base + n];
                        self.accum[n] += p * coeff;
                    }
                }
                for &v in &self.accum {
                    dst[offset] = Self::pack_u8(v);
                    offset += 1;
                }
            }
        }
    }

    /// Resize `src` image data into `dst`.
    pub fn resize(&mut self, src: &[u8], dst: &mut [u8]) {
        // TODO(Kagami):
        // * Multi-thread
        // * Bound checkings
        // * SIMD
        assert_eq!(src.len(), self.w1 * self.h1 * self.pix_fmt.get_size());
        assert_eq!(dst.len(), self.w2 * self.h2 * self.pix_fmt.get_size());
        self.sample_rows(src);
        self.sample_cols(dst)
    }
}

/// Create a new resizer instance. Alias for `Resizer::new`.
pub fn new(w1: usize, h1: usize, w2: usize, h2: usize, p: Pixel, t: Type) -> Resizer {
    Resizer::new(w1, h1, w2, h2, p, t)
}

/// Resize image data to the new dimension in a single step.
///
/// **NOTE:** If you need to resize to the same dimension multiple times,
/// consider creating an resizer instance since it's faster.
pub fn resize(
    w1: usize, h1: usize, w2: usize, h2: usize,
    p: Pixel, t: Type,
    src: &[u8], dst: &mut [u8],
) {
    Resizer::new(w1, h1, w2, h2, p, t).resize(src, dst)
}
