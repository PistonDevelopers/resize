use crate::formats;
pub use rgb::alt::Gray;
pub use rgb::RGB;
pub use rgb::RGBA;

/// Use [`Pixel`](crate::Pixel) presets to specify pixel format.
///
/// The trait represents a temporary object that adds pixels together.
pub trait PixelFormat {
    /// Pixel type in the source image
    type InputPixel: Copy;
    /// Pixel type in the destination image (usually the same as Input)
    type OutputPixel;
    /// Temporary struct for the pixel in floating-point
    type Accumulator: Copy;

    /// Create new floating-point pixel
    fn new() -> Self::Accumulator;
    /// Add new pixel with a given weight (first axis)
    fn add(&self, acc: &mut Self::Accumulator, inp: Self::InputPixel, coeff: f32);
    /// Add bunch of accumulated pixels with a weight (second axis)
    fn add_acc(acc: &mut Self::Accumulator, inp: Self::Accumulator, coeff: f32);
    /// Finalize, convert to output pixel format
    fn into_pixel(&self, acc: Self::Accumulator) -> Self::OutputPixel;
}

impl<F: ToFloat, T: ToFloat> PixelFormat for formats::Rgb<T, F> {
    type InputPixel = RGB<F>;
    type OutputPixel = RGB<T>;
    type Accumulator = RGB<f32>;

    #[inline(always)]
    fn new() -> Self::Accumulator {
        RGB::new(0.,0.,0.)
    }

    #[inline(always)]
    fn add(&self, acc: &mut Self::Accumulator, inp: RGB<F>, coeff: f32) {
        acc.r += inp.r.to_float() * coeff;
        acc.g += inp.g.to_float() * coeff;
        acc.b += inp.b.to_float() * coeff;
    }

    #[inline(always)]
    fn add_acc(acc: &mut Self::Accumulator, inp: Self::Accumulator, coeff: f32) {
        acc.r += inp.r * coeff;
        acc.g += inp.g * coeff;
        acc.b += inp.b * coeff;
    }

    #[inline(always)]
    fn into_pixel(&self, acc: Self::Accumulator) -> RGB<T> {
        RGB {
            r: T::from_float(acc.r),
            g: T::from_float(acc.g),
            b: T::from_float(acc.b),
        }
    }
}

impl<F: ToFloat, T: ToFloat> PixelFormat for formats::Rgba<T, F> {
    type InputPixel = RGBA<F>;
    type OutputPixel = RGBA<T>;
    type Accumulator = RGBA<f32>;

    #[inline(always)]
    fn new() -> Self::Accumulator {
        RGBA::new(0.,0.,0.,0.)
    }

    #[inline(always)]
    fn add(&self, acc: &mut Self::Accumulator, inp: RGBA<F>, coeff: f32) {
        acc.r += inp.r.to_float() * coeff;
        acc.g += inp.g.to_float() * coeff;
        acc.b += inp.b.to_float() * coeff;
        acc.a += inp.a.to_float() * coeff;
    }

    #[inline(always)]
    fn add_acc(acc: &mut Self::Accumulator, inp: Self::Accumulator, coeff: f32) {
        acc.r += inp.r * coeff;
        acc.g += inp.g * coeff;
        acc.b += inp.b * coeff;
        acc.a += inp.a * coeff;
    }

    #[inline(always)]
    fn into_pixel(&self, acc: Self::Accumulator) -> RGBA<T> {
        RGBA {
            r: T::from_float(acc.r),
            g: T::from_float(acc.g),
            b: T::from_float(acc.b),
            a: T::from_float(acc.a),
        }
    }
}

impl<F: ToFloat, T: ToFloat> PixelFormat for formats::RgbaPremultiply<T, F> {
    type InputPixel = RGBA<F>;
    type OutputPixel = RGBA<T>;
    type Accumulator = RGBA<f32>;

    #[inline(always)]
    fn new() -> Self::Accumulator {
        RGBA::new(0.,0.,0.,0.)
    }

    #[inline(always)]
    fn add(&self, acc: &mut Self::Accumulator, inp: RGBA<F>, coeff: f32) {
        let a_coeff = inp.a.to_float() * coeff;
        acc.r += inp.r.to_float() * a_coeff;
        acc.g += inp.g.to_float() * a_coeff;
        acc.b += inp.b.to_float() * a_coeff;
        acc.a += a_coeff;
    }

    #[inline(always)]
    fn add_acc(acc: &mut Self::Accumulator, inp: Self::Accumulator, coeff: f32) {
        acc.r += inp.r * coeff;
        acc.g += inp.g * coeff;
        acc.b += inp.b * coeff;
        acc.a += inp.a * coeff;
    }

    #[inline(always)]
    fn into_pixel(&self, acc: Self::Accumulator) -> RGBA<T> {
        if acc.a > 0. {
            let inv = 1.0 / acc.a;
            RGBA {
                r: T::from_float(acc.r * inv),
                g: T::from_float(acc.g * inv),
                b: T::from_float(acc.b * inv),
                a: T::from_float(acc.a),
            }
        } else {
            let zero = T::from_float(0.);
            RGBA::new(zero, zero, zero, zero)
        }
    }
}

impl<F: ToFloat, T: ToFloat> PixelFormat for formats::Gray<F, T> {
    type InputPixel = Gray<F>;
    type OutputPixel = Gray<T>;
    type Accumulator = Gray<f32>;

    #[inline(always)]
    fn new() -> Self::Accumulator {
        Gray::new(0.)
    }

    #[inline(always)]
    fn add(&self, acc: &mut Self::Accumulator, inp: Gray<F>, coeff: f32) {
        acc.0 += inp.0.to_float() * coeff;
    }

    #[inline(always)]
    fn add_acc(acc: &mut Self::Accumulator, inp: Self::Accumulator, coeff: f32) {
        acc.0 += inp.0 * coeff;
    }

    #[inline(always)]
    fn into_pixel(&self, acc: Self::Accumulator) -> Gray<T> {
        Gray::new(T::from_float(acc.0))
    }
}

use self::f::ToFloat;
mod f {
    /// Internal, please don't use
    pub trait ToFloat: Sized + Copy + 'static {
        fn to_float(self) -> f32;
        fn from_float(f: f32) -> Self;
    }

    impl ToFloat for u8 {
        #[inline(always)]
        fn to_float(self) -> f32 {
            self as f32
        }

        #[inline(always)]
        fn from_float(f: f32) -> Self {
            (f + 0.5) as u8
        }
    }

    impl ToFloat for u16 {
        #[inline(always)]
        fn to_float(self) -> f32 {
            self as f32
        }

        #[inline(always)]
        fn from_float(f: f32) -> Self {
            (f + 0.5) as u16
        }
    }

    impl ToFloat for f32 {
        #[inline(always)]
        fn to_float(self) -> f32 {
            self
        }

        #[inline(always)]
        fn from_float(f: f32) -> Self {
            f
        }
    }

    impl ToFloat for f64 {
        #[inline(always)]
        fn to_float(self) -> f32 {
            self as f32
        }

        #[inline(always)]
        fn from_float(f: f32) -> Self {
            f as f64
        }
    }
}
