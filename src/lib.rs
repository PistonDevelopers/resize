//! Simple resampling library in pure Rust.
//!
//! # Examples
//!
//! ```
//! use resize::Pixel::RGB8;
//! use resize::Type::Lanczos3;
//! use rgb::RGB8;
//! use rgb::FromSlice;
//!
//! // Downscale by 2x.
//! let (w1, h1) = (640, 480);
//! let (w2, h2) = (320, 240);
//! // Don't forget to fill `src` with image data (RGB8).
//! let src = vec![0;w1*h1*3];
//! // Destination buffer. Must be mutable.
//! let mut dst = vec![0;w2*h2*3];
//! // Create reusable instance.
//! let mut resizer = resize::new(w1, h1, w2, h2, RGB8, Lanczos3)?;
//! // Do resize without heap allocations.
//! // Might be executed multiple times for different `src` or `dst`.
//! resizer.resize(src.as_rgb(), dst.as_rgb_mut());
//! # Ok::<_, resize::Error>(())
//! ```
// Current implementation is based on:
// * https://github.com/sekrit-twc/zimg/tree/master/src/zimg/resize
// * https://github.com/PistonDevelopers/image/blob/master/src/imageops/sample.rs
#![deny(missing_docs)]

use std::collections::HashMap;
use std::sync::Arc;
use std::f32;
use std::fmt;
use std::num::NonZeroUsize;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// See [Error]
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Pixel format from the [rgb] crate.
pub mod px;
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
    #[inline(always)]
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

/// Predefined constants for supported pixel formats.
#[allow(non_snake_case)]
#[allow(non_upper_case_globals)]
pub mod Pixel {
    use std::marker::PhantomData;
    use crate::formats;

    /// Grayscale, 8-bit.
    #[cfg_attr(docsrs, doc(alias = "Grey"))]
    pub const Gray8: formats::Gray<u8, u8> = formats::Gray(PhantomData);
    /// Grayscale, 16-bit, native endian.
    pub const Gray16: formats::Gray<u16, u16> = formats::Gray(PhantomData);

    /// Grayscale, 32-bit float
    pub const GrayF32: formats::Gray<f32, f32> = formats::Gray(PhantomData);
    /// Grayscale, 64-bit float
    pub const GrayF64: formats::Gray<f64, f64> = formats::Gray(PhantomData);

    /// RGB, 8-bit per component.
    #[cfg_attr(docsrs, doc(alias = "RGB24"))]
    pub const RGB8: formats::Rgb<u8, u8> = formats::Rgb(PhantomData);
    /// RGB, 16-bit per component, native endian.
    #[cfg_attr(docsrs, doc(alias = "RGB48"))]
    pub const RGB16: formats::Rgb<u16, u16> = formats::Rgb(PhantomData);
    /// RGBA, 8-bit per component. Components are scaled independently. Use this if the input is already alpha-premultiplied.
    ///
    /// Preserves RGB values of fully-transparent pixels. Expect halos around edges of transparency if using regular, uncorrelated RGBA. See [RGBA8P].
    #[cfg_attr(docsrs, doc(alias = "RGBA32"))]
    pub const RGBA8: formats::Rgba<u8, u8> = formats::Rgba(PhantomData);
    /// RGBA, 16-bit per component, native endian. Components are scaled independently. Use this if the input is already alpha-premultiplied.
    ///
    /// Preserves RGB values of fully-transparent pixels. Expect halos around edges of transparency if using regular, uncorrelated RGBA. See [RGBA16P].
    #[cfg_attr(docsrs, doc(alias = "RGBA64"))]
    pub const RGBA16: formats::Rgba<u16, u16> = formats::Rgba(PhantomData);
    /// RGBA, 8-bit per component. RGB components will be converted to premultiplied during scaling, and then converted back to uncorrelated.
    ///
    /// Clears "dirty alpha". Use this for high-quality scaling of regular uncorrelated (not premultiplied) RGBA bitmaps.
    #[cfg_attr(docsrs, doc(alias = "premultiplied"))]
    #[cfg_attr(docsrs, doc(alias = "prem"))]
    pub const RGBA8P: formats::RgbaPremultiply<u8, u8> = formats::RgbaPremultiply(PhantomData);
    /// RGBA, 16-bit per component, native endian. RGB components will be converted to premultiplied during scaling, and then converted back to uncorrelated.
    ///
    /// Clears "dirty alpha". Use this for high-quality scaling of regular uncorrelated (not premultiplied) RGBA bitmaps.
    pub const RGBA16P: formats::RgbaPremultiply<u16, u16> = formats::RgbaPremultiply(PhantomData);

    /// RGB, 32-bit float per component. This is pretty efficient, since resizing uses f32 internally.
    pub const RGBF32: formats::Rgb<f32, f32> = formats::Rgb(PhantomData);
    /// RGB, 64-bit double per component.
    pub const RGBF64: formats::Rgb<f64, f64> = formats::Rgb(PhantomData);

    /// RGBA, 32-bit float per component. This is pretty efficient, since resizing uses f32 internally.
    ///
    /// Components are scaled independently (no premultiplication applied)
    pub const RGBAF32: formats::Rgba<f32, f32> = formats::Rgba(PhantomData);
    /// RGBA, 64-bit double per component.
    ///
    /// Components are scaled independently (no premultiplication applied)
    pub const RGBAF64: formats::Rgba<f64, f64> = formats::Rgba(PhantomData);
}

/// Implementation detail
///
/// These structs implement `PixelFormat` trait that allows conversion to and from internal pixel representation.
#[doc(hidden)]
pub mod formats {
    use std::marker::PhantomData;
    /// RGB pixels
    #[derive(Debug, Copy, Clone)]
    pub struct Rgb<InputSubpixel, OutputSubpixel>(pub(crate) PhantomData<(InputSubpixel, OutputSubpixel)>);
    /// RGBA pixels, each channel is independent. Compatible with premultiplied input/output.
    #[derive(Debug, Copy, Clone)]
    pub struct Rgba<InputSubpixel, OutputSubpixel>(pub(crate) PhantomData<(InputSubpixel, OutputSubpixel)>);
    /// Apply premultiplication to RGBA pixels during scaling. Assumes **non**-premultiplied input/output.
    #[derive(Debug, Copy, Clone)]
    pub struct RgbaPremultiply<InputSubpixel, OutputSubpixel>(pub(crate) PhantomData<(InputSubpixel, OutputSubpixel)>);
    /// Grayscale pixels
    #[derive(Debug, Copy, Clone)]
    pub struct Gray<InputSubpixel, OutputSubpixel>(pub(crate) PhantomData<(InputSubpixel, OutputSubpixel)>);
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
    w1: NonZeroUsize,
    h1: NonZeroUsize,
    /// Vec's len == target dimensions
    coeffs_w: Vec<CoeffsLine>,
    coeffs_h: Vec<CoeffsLine>,
}

impl Scale {
    #[inline(always)]
    fn w2(&self) -> usize {
        self.coeffs_w.len()
    }

    #[inline(always)]
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
    pub fn new(source_width: usize, source_heigth: usize, dest_width: usize, dest_height: usize, filter_type: Type) -> Result<Self> {
        let source_width = NonZeroUsize::new(source_width).ok_or(Error::InvalidParameters)?;
        let source_heigth = NonZeroUsize::new(source_heigth).ok_or(Error::InvalidParameters)?;
        if dest_width == 0 || dest_height == 0 {
            return Err(Error::InvalidParameters);
        }
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
        recycled_coeffs.try_reserve(dest_width.max(dest_height))?;

        let coeffs_w = Self::calc_coeffs(source_width, dest_width, filter, &mut recycled_coeffs)?;
        let coeffs_h = if source_heigth == source_width && dest_height == dest_width {
            coeffs_w.clone()
        } else {
            Self::calc_coeffs(source_heigth, dest_height, filter, &mut recycled_coeffs)?
        };

        Ok(Self {
            w1: source_width,
            h1: source_heigth,
            coeffs_w,
            coeffs_h,
        })
    }

    fn calc_coeffs(s1: NonZeroUsize, s2: usize, (kernel, support): (&dyn Fn(f32) -> f32, f32), recycled_coeffs: &mut HashMap<(usize, [u8; 4], [u8; 4]), Arc<[f32]>>) -> Result<Vec<CoeffsLine>> {
        let ratio = s1.get() as f64 / s2 as f64;
        // Scale the filter when downsampling.
        let filter_scale = ratio.max(1.);
        let filter_radius = (support as f64 * filter_scale).ceil();
        let mut res = Vec::new();
        res.try_reserve_exact(s2)?;
        for x2 in 0..s2 {
            let x1 = (x2 as f64 + 0.5) * ratio - 0.5;
            let start = (x1 - filter_radius).ceil() as isize;
            let start = start.min(s1.get() as isize - 1).max(0) as usize;
            let end = (x1 + filter_radius).floor() as isize;
            let end = (end.min(s1.get() as isize - 1).max(0) as usize).max(start);
            let sum: f64 = (start..=end).map(|i| (kernel)(((i as f64 - x1) / filter_scale) as f32) as f64).sum();
            let key = (end - start, (filter_scale as f32).to_ne_bytes(), (start as f32 - x1 as f32).to_ne_bytes());
            let coeffs = if let Some(k) = recycled_coeffs.get(&key) { k.clone() } else {
                let tmp = (start..=end).map(|i| {
                    let n = ((i as f64 - x1) / filter_scale) as f32;
                    ((kernel)(n.min(support).max(-support)) as f64 / sum) as f32
                }).collect::<Arc<[_]>>();
                recycled_coeffs.try_reserve(1)?;
                recycled_coeffs.insert(key, tmp.clone());
                tmp
            };
            res.push(CoeffsLine { start, coeffs });
        }
        Ok(res)
    }
}

impl<Format: PixelFormat> Resizer<Format> {
    /// Create a new resizer instance.
    #[inline]
    pub fn new(source_width: usize, source_heigth: usize, dest_width: usize, dest_height: usize, pixel_format: Format, filter_type: Type) -> Result<Self> {
        Ok(Self {
            scale: Scale::new(source_width, source_heigth, dest_width, dest_height, filter_type)?,
            tmp: Vec::new(),
            pix_fmt: pixel_format,
        })
    }

    /// Stride is a length of the source row (>= W1)
    #[cfg(not(feature = "rayon"))]
    fn resample_both_axes(&mut self, src: &[Format::InputPixel], stride: NonZeroUsize, mut dst: &mut [Format::OutputPixel]) -> Result<()> {
        self.tmp.clear();
        self.tmp.try_reserve(self.scale.w2() * self.scale.h1.get())?;

        // Outer loop resamples W2xH1 to W2xH2
        let mut src_rows = src.chunks(stride.get());
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
        Ok(())
    }

    #[cfg(feature = "rayon")]
    fn resample_both_axes(&mut self, src: &[Format::InputPixel], stride: NonZeroUsize, dst: &mut [Format::OutputPixel]) -> Result<()> {
        use std::sync::atomic::AtomicPtr;

        let stride = stride.get();
        let pix_fmt = &self.pix_fmt;
        let w2 = self.scale.w2();
        let h2 = self.scale.h2();

        // Ensure the destination buffer has adequate size for the resampling operation.
        if dst.len() < w2 * h2 {
            return Err(Error::InvalidParameters);
        }

        // Prepare the temporary buffer for intermediate storage.
        self.tmp.clear();
        self.tmp.try_reserve(self.scale.w2() * self.scale.h1.get())?;

        // Initialize an atomic pointer for safe multithreaded access.
        let tmp_ptr = AtomicPtr::new(self.tmp.as_mut_ptr());

        // Horizontal Resampling
        // Process each row in parallel. Each pixel within a row is processed sequentially.
        src.par_chunks(stride).enumerate().for_each(|(y, row)| {
            // For each pixel in the row, calculate the horizontal resampling and store the result.
            self.scale.coeffs_w.par_iter().enumerate().for_each(|(x, col)| {
                // Acquire a safe reference to the current position in the temporary buffer.
                let tmp_raw_ptr = tmp_ptr.load(std::sync::atomic::Ordering::Relaxed);

                let mut accum = Format::new();
                let in_px = &row[col.start..col.start + col.coeffs.len()];
                for (coeff, in_px) in col.coeffs.iter().copied().zip(in_px.iter().copied()) {
                    pix_fmt.add(&mut accum, in_px, coeff);
                }

                // Determine the location in the temporary buffer to store the result.
                let pixel_offset = y * self.scale.coeffs_w.len() + x;
                // Write the accumulated value to the temporary buffer.
                unsafe {
                    *tmp_raw_ptr.add(pixel_offset) = accum;
                }
            });
        });

        // Vertical Resampling
        // Process each row in parallel. Each pixel within a row is processed sequentially.
        let dst_ptr = AtomicPtr::new(dst.as_mut_ptr());
        self.scale.coeffs_h.par_iter().enumerate().for_each(|(y, row)| {
            // For each pixel in the row, calculate the vertical resampling and store the result directly into the destination buffer.
             (0..w2).into_par_iter().for_each(|x| {
                // Acquire a safe reference to the current position in the temporary buffer.
                let tmp_raw_ptr = tmp_ptr.load(std::sync::atomic::Ordering::Relaxed);

                // Determine the start of the current row in the temporary buffer.
                let tmp_row_start = unsafe { tmp_raw_ptr.add(w2 * row.start) };

                let mut accum = Format::new();
                for (coeff_idx, coeff) in row.coeffs.iter().copied().enumerate() {
                    // Calculate the appropriate pixel location based on the coefficient index.
                    let other_row = unsafe { tmp_row_start.add(coeff_idx * w2) };
                    let other_pixel = unsafe { *other_row.add(x) };
                    Format::add_acc(&mut accum, other_pixel, coeff);
                }

                // Determine the location in the destination buffer to store the result.
                let raw_ptr = dst_ptr.load(std::sync::atomic::Ordering::Relaxed);
                // Write the accumulated value to the destination buffer.
                unsafe {
                    *raw_ptr.add(y * w2 + x) = pix_fmt.into_pixel(accum);
                }
            });
        });

        Ok(())
    }

    /// Resize `src` image data into `dst`.
    #[inline]
    pub(crate) fn resize_internal(&mut self, src: &[Format::InputPixel], src_stride: NonZeroUsize, dst: &mut [Format::OutputPixel]) -> Result<()> {
        if self.scale.w1.get() > src_stride.get() ||
            src.len() < (src_stride.get() * self.scale.h1.get()) + self.scale.w1.get() - src_stride.get() ||
            dst.len() != self.scale.w2() * self.scale.h2() {
                return Err(Error::InvalidParameters)
            }
        self.resample_both_axes(src, src_stride, dst)
    }
}

impl<Format: PixelFormat> Resizer<Format> {
    /// Resize `src` image data into `dst`.
    #[inline]
    pub fn resize(&mut self, src: &[Format::InputPixel], dst: &mut [Format::OutputPixel]) -> Result<()> {
        self.resize_internal(src, self.scale.w1, dst)
    }

    /// Resize `src` image data into `dst`, skipping `stride` pixels each row.
    #[inline]
    pub fn resize_stride(&mut self, src: &[Format::InputPixel], src_stride: usize, dst: &mut [Format::OutputPixel]) -> Result<()> {
        let src_stride = NonZeroUsize::new(src_stride).ok_or(Error::InvalidParameters)?;
        self.resize_internal(src, src_stride, dst)
    }
}

/// Create a new resizer instance. Alias for `Resizer::new`.
#[inline(always)]
pub fn new<Format: PixelFormat>(src_width: usize, src_height: usize, dest_width: usize, dest_height: usize, pixel_format: Format, filter_type: Type) -> Result<Resizer<Format>> {
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
pub fn resize<Format: PixelFormat>(
    src_width: usize, src_height: usize, dest_width: usize, dest_height: usize,
    pixel_format: Format, filter_type: Type,
    src: &[Format::InputPixel], dst: &mut [Format::OutputPixel],
) -> Result<()> {
    Resizer::<Format>::new(src_width, src_height, dest_width, dest_height, pixel_format, filter_type)?.resize(src, dst)
}

/// Resizing may run out of memory
#[derive(Debug)]
pub enum Error {
    /// Allocation failed
    OutOfMemory,
    /// e.g. width or height can't be 0
    InvalidParameters,
}

impl std::error::Error for Error {}

impl From<std::collections::TryReserveError> for Error {
    #[inline(always)]
    fn from(_: std::collections::TryReserveError) -> Self {
        Self::OutOfMemory
    }
}

impl fmt::Display for Error {
    #[cold]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(match self {
            Self::OutOfMemory => "out of memory",
            Self::InvalidParameters => "invalid parameters"
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oom() {
        let _ = new(2, 2, isize::max_value() as _, isize::max_value() as _, Pixel::Gray16, Type::Triangle);
    }

    #[test]
    fn niche() {
        assert_eq!(std::mem::size_of::<Resizer<formats::Gray<f32, f32>>>(), std::mem::size_of::<Option<Resizer<formats::Gray<f32, f32>>>>());
    }

    #[test]
    fn zeros() {
        assert!(new(1, 1, 1, 0, Pixel::Gray16, Type::Triangle).is_err());
        assert!(new(1, 1, 0, 1, Pixel::Gray8, Type::Catrom).is_err());
        assert!(new(1, 0, 1, 1, Pixel::RGBAF32, Type::Lanczos3).is_err());
        assert!(new(0, 1, 1, 1, Pixel::RGB8, Type::Mitchell).is_err());
    }

    #[test]
    fn premultiply() {
        use px::RGBA;
        let mut r = new(2, 2, 3, 4, Pixel::RGBA8P, Type::Triangle).unwrap();
        let mut dst = vec![RGBA::new(0u8,0,0,0u8); 12];
        r.resize(&[
            RGBA::new(255,127,3,255), RGBA::new(0,0,0,0),
            RGBA::new(255,255,255,0), RGBA::new(0,255,255,0),
        ], &mut dst).unwrap();
        assert_eq!(&dst, &[
            RGBA { r: 255, g: 127, b: 3, a: 255 }, RGBA { r: 255, g: 127, b: 3, a: 128 }, RGBA { r: 0, g: 0, b: 0, a: 0 },
            RGBA { r: 255, g: 127, b: 3, a: 191 }, RGBA { r: 255, g: 127, b: 3, a: 96 }, RGBA { r: 0, g: 0, b: 0, a: 0 },
            RGBA { r: 255, g: 127, b: 3, a: 64 }, RGBA { r: 255, g: 127, b: 3, a: 32 }, RGBA { r: 0, g: 0, b: 0, a: 0 },
            RGBA { r: 0, g: 0, b: 0, a: 0 }, RGBA { r: 0, g: 0, b: 0, a: 0 }, RGBA { r: 0, g: 0, b: 0, a: 0 }
        ]);
    }

    #[test]
    fn premultiply_solid() {
        use px::RGBA;
        let mut r = new(2, 2, 3, 4, Pixel::RGBA8P, Type::Triangle).unwrap();
        let mut dst = vec![RGBA::new(0u8,0,0,0u8); 12];
        r.resize(&[
            RGBA::new(255,255,255,255), RGBA::new(0,0,0,255),
            RGBA::new(0,0,0,255), RGBA::new(0,0,0,255),
        ], &mut dst).unwrap();
        assert_eq!(&dst, &[
            RGBA { r: 255, g: 255, b: 255, a: 255 }, RGBA { r: 128, g: 128, b: 128, a: 255 }, RGBA { r: 0, g: 0, b: 0, a: 255 },
            RGBA { r: 191, g: 191, b: 191, a: 255 }, RGBA { r: 96, g: 96, b: 96, a: 255 }, RGBA { r: 0, g: 0, b: 0, a: 255 },
            RGBA { r: 64, g: 64, b: 64, a: 255 }, RGBA { r: 32, g: 32, b: 32, a: 255 }, RGBA { r: 0, g: 0, b: 0, a: 255 },
            RGBA { r: 0, g: 0, b: 0, a: 255 }, RGBA { r: 0, g: 0, b: 0, a: 255 }, RGBA { r: 0, g: 0, b: 0, a: 255 },
        ]);
    }

    #[test]
    fn resize_stride() {
        use rgb::FromSlice;

        let mut r = new(2, 2, 3, 4, Pixel::Gray16, Type::Triangle).unwrap();
        let mut dst = vec![0; 12];
        r.resize_stride(&[
            65535,65535,1,2,
            65535,65535,3,4,
        ].as_gray(), 4, dst.as_gray_mut()).unwrap();
        assert_eq!(&dst, &[65535; 12]);
    }

    #[test]
    fn resize_float() {
        use rgb::FromSlice;

        let mut r = new(2, 2, 3, 4, Pixel::GrayF32, Type::Triangle).unwrap();
        let mut dst = vec![0.; 12];
        r.resize_stride(&[
            65535.,65535.,1.,2.,
            65535.,65535.,3.,4.,
        ].as_gray(), 4, dst.as_gray_mut()).unwrap();
        assert_eq!(&dst, &[65535.; 12]);
    }
}
