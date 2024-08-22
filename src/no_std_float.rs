/// Alternative basic float operations for no_std
pub(crate) trait FloatExt {
    fn floor(self) -> Self;
    fn ceil(self) -> Self;
    fn sqrt(self) -> Self;
    fn round(self) -> Self;
    fn abs(self) -> Self;
    fn trunc(self) -> Self;
    fn fract(self) -> Self;
    fn sin(self) -> Self;
    fn exp(self) -> Self;
    fn powi(self, n: i32) -> Self;
}

impl FloatExt for f32 {
    #[inline]
    fn floor(self) -> Self {
        libm::floorf(self)
    }
    #[inline]
    fn ceil(self) -> Self {
        libm::ceilf(self)
    }
    #[inline]
    fn sqrt(self) -> Self {
        libm::sqrtf(self)
    }
    #[inline]
    fn round(self) -> Self {
        libm::roundf(self)
    }
    #[inline]
    fn abs(self) -> Self {
        libm::fabsf(self)
    }
    #[inline]
    fn trunc(self) -> Self {
        libm::truncf(self)
    }
    #[inline]
    fn fract(self) -> Self {
        self - self.trunc()
    }
    #[inline]
    fn sin(self) -> Self {
        libm::sinf(self)
    }
    #[inline]
    fn powi(self, n: i32) -> Self {
        libm::powf(self, n as _)
    }
    #[inline]
    fn exp(self) -> Self {
        libm::expf(self)
    }
}

impl FloatExt for f64 {
    #[inline]
    fn floor(self) -> Self {
        libm::floor(self)
    }
    #[inline]
    fn ceil(self) -> Self {
        libm::ceil(self)
    }
    #[inline]
    fn sqrt(self) -> Self {
        libm::sqrt(self)
    }
    #[inline]
    fn round(self) -> Self {
        libm::round(self)
    }
    #[inline]
    fn abs(self) -> Self {
        libm::fabs(self)
    }
    #[inline]
    fn trunc(self) -> Self {
        libm::trunc(self)
    }
    #[inline]
    fn fract(self) -> Self {
        self - self.trunc()
    }
    #[inline]
    fn sin(self) -> Self {
        libm::sin(self)
    }
    #[inline]
    fn powi(self, n: i32) -> Self {
        libm::pow(self, n as _)
    }
    #[inline]
    fn exp(self) -> Self {
        libm::exp(self)
    }
}
