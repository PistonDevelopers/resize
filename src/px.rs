use std::mem;
use crate::Pixel;

/// See `Pixel`
pub trait PixelFormat {
    /// Array to hold temporary values.
    type Accumulator: AsRef<[f32]> + AsMut<[f32]>;
    /// Type of a Subpixel of each pixel (8 or 16 bits).
    type Subpixel: Copy;

    /// New empty Accumulator.
    fn new_accum() -> Self::Accumulator;

    /// Convert float to integer value in range appropriate for this pixel format.
    fn into_subpixel(v: f32) -> Self::Subpixel;

    /// Convert pixel component to float
    fn from_subpixel(v: &Self::Subpixel) -> f32;

    /// Size of one pixel in that format in bytes.
    #[inline(always)]
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
    type Accumulator = [f32; 1];
    type Subpixel = u8;

    #[must_use]
    #[inline(always)]
    fn new_accum() -> Self::Accumulator {
        [0.0; 1]
    }

    #[must_use]
    #[inline(always)]
    fn into_subpixel(v: f32) -> Self::Subpixel {
        pack_u8(v)
    }

    #[inline(always)]
    fn from_subpixel(px: &Self::Subpixel) -> f32 {
        *px as f32
    }
}

impl PixelFormat for Pixel::Gray16 {
    type Accumulator = [f32; 1];
    type Subpixel = u16;

    #[must_use]
    #[inline(always)]
    fn new_accum() -> Self::Accumulator {
        [0.0; 1]
    }

    #[must_use]
    #[inline(always)]
    fn into_subpixel(v: f32) -> Self::Subpixel {
        pack_u16(v)
    }

    #[inline(always)]
    fn from_subpixel(px: &Self::Subpixel) -> f32 {
        *px as f32
    }
}

impl PixelFormat for Pixel::RGB24 {
    type Accumulator = [f32; 3];
    type Subpixel = u8;

    #[must_use]
    #[inline(always)]
    fn new_accum() -> Self::Accumulator {
        [0.0; 3]
    }

    #[must_use]
    #[inline(always)]
    fn into_subpixel(v: f32) -> Self::Subpixel {
        pack_u8(v)
    }

    #[inline(always)]
    fn from_subpixel(px: &Self::Subpixel) -> f32 {
        *px as f32
    }
}
impl PixelFormat for Pixel::RGBA {
    type Accumulator = [f32; 4];
    type Subpixel = u8;

    #[must_use]
    #[inline(always)]
    fn new_accum() -> Self::Accumulator {
        [0.0; 4]
    }

    #[must_use]
    #[inline(always)]
    fn into_subpixel(v: f32) -> Self::Subpixel {
        pack_u8(v)
    }

    #[inline(always)]
    fn from_subpixel(px: &Self::Subpixel) -> f32 {
        *px as f32
    }
}
impl PixelFormat for Pixel::RGB48 {
    type Accumulator = [f32; 3];
    type Subpixel = u16;

    #[must_use]
    #[inline(always)]
    fn new_accum() -> Self::Accumulator {
        [0.0; 3]
    }

    #[must_use]
    #[inline(always)]
    fn into_subpixel(v: f32) -> Self::Subpixel {
        pack_u16(v)
    }

    #[inline(always)]
    fn from_subpixel(px: &Self::Subpixel) -> f32 {
        *px as f32
    }
}
impl PixelFormat for Pixel::RGBA64 {
    type Accumulator = [f32; 4];
    type Subpixel = u16;

    #[must_use]
    #[inline(always)]
    fn new_accum() -> Self::Accumulator {
        [0.0; 4]
    }

    #[must_use]
    #[inline(always)]
    fn into_subpixel(v: f32) -> Self::Subpixel {
        pack_u16(v)
    }

    #[inline(always)]
    fn from_subpixel(px: &Self::Subpixel) -> f32 {
        *px as f32
    }
}


#[inline]
fn pack_u8(v: f32) -> u8 {
    if v > 255.0 {
        255
    } else if v < 0.0 {
        0
    } else {
        v.round() as u8
    }
}

#[inline]
fn pack_u16(v: f32) -> u16 {
    if v > 65535.0 {
        65535
    } else if v < 0.0 {
        0
    } else {
        v.round() as u16
    }
}
