//! Simple resampling library in pure Rust.
#![deny(missing_docs)]

use std::f32;

/// Resizing type to use.
pub enum Type {
    /// Triangle (bilinear) filter.
    Triangle,
    /// Sinc-windowed filter with radius of 3.
    Lanczos3,
    /// Custom filter.
    Custom(Filter),
}

/// Resampling filter.
pub struct Filter {
    /// Filter kernel.
    pub kernel: Box<Fn(f32) -> f32>,
    /// Filter support.
    pub support: f32,
}

#[inline]
fn triangle_kernel(x: f32) -> f32 {
    f32::max(1.0 - x.abs(), 0.0)
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
fn lanczos3_kernel(x: f32) -> f32 {
    if x.abs() < 3.0 {
        sinc(x) * sinc(x / 3.0)
    } else {
        0.0
    }
}

/// Simple resampler with preallocated buffers and coeffecients for the given
/// dimensions. See also:
/// * https://github.com/sekrit-twc/zimg/tree/master/src/zimg/resize
/// * https://github.com/PistonDevelopers/image/blob/master/src/imageops/sample.rs
#[derive(Debug)]
pub struct Resizer {
    // Source/target dimensions.
    w1: usize,
    h1: usize,
    w2: usize,
    h2: usize,
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

impl Resizer {
    /// Create a new resizer instance for the given dimensions and filter.
    pub fn new(w1: usize, h1: usize, w2: usize, h2: usize, t: Type) -> Resizer {
        let filter = match t {
            Type::Triangle => Filter {
                kernel: Box::new(triangle_kernel),
                support: 1.0,
            },
            Type::Lanczos3 => Filter {
                kernel: Box::new(lanczos3_kernel),
                support: 3.0,
            },
            Type::Custom(f) => f,
        };
        Resizer {
            w1: w1,
            h1: h1,
            w2: w2,
            h2: h2,
            tmp: vec![0.0;w1*h2],
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
    fn pack_u8(v: f32) -> u8 {
        let mut v = v.round();
        v = f32::min(f32::max(v, 0.0), 255.0);
        v as u8
    }

    /// Resample W1xH1 to W1xH2.
    fn sample_rows(&mut self, src: &[u8]) {
        // FIXME(Kagami): Avoid bound checkings.
        let mut offset = 0;
        for x1 in 0..self.w1 {
            for y2 in 0..self.h2 {
                let ref line = self.coeffs_h[y2];
                let mut accum = 0.0;
                for (i, coeff) in line.data.iter().enumerate() {
                    let y0 = line.left + i;
                    let p = src[y0*self.w1 + x1] as f32;
                    accum += p * coeff;
                }
                self.tmp[offset] = accum;
                offset += 1;
            }
        }
    }

    /// Resample W1xH2 to W2xH2.
    fn sample_cols(&self, dst: &mut [u8]) {
        let mut offset = 0;
        for y2 in 0..self.h2 {
            for x2 in 0..self.w2 {
                let ref line = self.coeffs_w[x2];
                let mut accum = 0.0;
                for (i, coeff) in line.data.iter().enumerate() {
                    let x0 = line.left + i;
                    let p = self.tmp[x0*self.h2 + y2];
                    accum += p * coeff;
                }
                dst[offset] = Self::pack_u8(accum);
                offset += 1;
            }
        }
    }

    /// Resize `src` image data into `dst`.
    pub fn run(&mut self, src: &[u8], dst: &mut [u8]) {
        assert_eq!(src.len(), self.w1 * self.h1);
        assert_eq!(dst.len(), self.w2 * self.h2);
        self.sample_rows(src);
        self.sample_cols(dst)
    }
}

/// Create a new resizer instance. Alias for `Resizer::new`.
pub fn new(w1: usize, h1: usize, w2: usize, h2: usize, t: Type) -> Resizer {
    Resizer::new(w1, h1, w2, h2, t)
}
