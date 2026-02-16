// SPDX-License-Identifier: LGPL-3.0-or-later
// Ported from lsp-plugins/lsp-dsp-lib (C++)

//! Core data types for the DSP library.
//!
//! These types mirror the C++ originals from `lsp-dsp-lib` to ensure
//! algorithmic compatibility during the port.

/// Amplification threshold below which signals are considered silent.
pub const AMPLIFICATION_THRESH: f32 = 1e-8;

// ─── Biquad filter types ───────────────────────────────────────────────────

/// Number of delay (memory) elements in a biquad filter.
pub const BIQUAD_D_ITEMS: usize = 16;

/// Coefficients for a single biquad filter section.
///
/// Implements the difference equation:
/// ```text
///   y[n] = b0*x[n] + d0
///   d0   = b1*x[n] - a1*y[n] + d1
///   d1   = b2*x[n] - a2*y[n]
/// ```
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct BiquadX1 {
    pub b0: f32,
    pub b1: f32,
    pub b2: f32,
    pub a1: f32,
    pub a2: f32,
    pub _pad: [f32; 3],
}

/// Coefficients for two parallel biquad filter sections.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct BiquadX2 {
    pub b0: [f32; 2],
    pub b1: [f32; 2],
    pub b2: [f32; 2],
    pub a1: [f32; 2],
    pub a2: [f32; 2],
    pub _pad: [f32; 2],
}

/// Coefficients for four parallel biquad filter sections.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct BiquadX4 {
    pub b0: [f32; 4],
    pub b1: [f32; 4],
    pub b2: [f32; 4],
    pub a1: [f32; 4],
    pub a2: [f32; 4],
}

/// Coefficients for eight parallel biquad filter sections.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct BiquadX8 {
    pub b0: [f32; 8],
    pub b1: [f32; 8],
    pub b2: [f32; 8],
    pub a1: [f32; 8],
    pub a2: [f32; 8],
}

/// Biquad filter bank with coefficient storage and delay memory.
///
/// The coefficients union can be accessed as x1, x2, x4, or x8 depending
/// on how many parallel sections are needed. The `d` array stores the
/// filter's internal delay state.
#[derive(Debug, Clone)]
pub struct Biquad {
    /// Delay memory elements.
    pub d: [f32; BIQUAD_D_ITEMS],
    /// Filter coefficients.
    pub coeffs: BiquadCoeffs,
}

/// Biquad coefficient storage — one of the parallel configurations.
#[derive(Debug, Clone, Copy)]
pub enum BiquadCoeffs {
    X1(BiquadX1),
    X2(BiquadX2),
    X4(BiquadX4),
    X8(BiquadX8),
}

impl Default for Biquad {
    fn default() -> Self {
        Self {
            d: [0.0; BIQUAD_D_ITEMS],
            coeffs: BiquadCoeffs::X1(BiquadX1::default()),
        }
    }
}

impl Biquad {
    /// Reset the delay memory to zero (clear filter state).
    pub fn reset(&mut self) {
        self.d = [0.0; BIQUAD_D_ITEMS];
    }
}

// ─── Analog filter cascade ─────────────────────────────────────────────────

/// Analog filter cascade transfer function representation.
///
/// Represents a single second-order section with numerator (`t`) and
/// denominator (`b`) polynomial coefficients.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct FilterCascade {
    /// Numerator coefficients (zeros): `t[0] + t[1]*s + t[2]*s^2 + t[3]*s^3`
    pub t: [f32; 4],
    /// Denominator coefficients (poles): `b[0] + b[1]*s + b[2]*s^2 + b[3]*s^3`
    pub b: [f32; 4],
}

// ─── Dynamics processing types ─────────────────────────────────────────────

/// Compressor knee model with three zones:
/// 1. Below `start`: constant gain amplification
/// 2. Between `start` and `end`: soft knee (Hermite polynomial)
/// 3. Above `end`: linear gain reduction (in log domain)
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct CompressorKnee {
    /// Knee start point (gain units, linear).
    pub start: f32,
    /// Knee end point (gain units, linear).
    pub end: f32,
    /// Pre-amplification factor (makeup gain).
    pub gain: f32,
    /// Hermite interpolation coefficients for the soft knee (quadratic in log domain).
    pub herm: [f32; 3],
    /// Linear compression parameters post-knee: `tilt[0]*ln(x) + tilt[1]`.
    pub tilt: [f32; 2],
}

/// Dual-knee compressor combining two [`CompressorKnee`] through
/// multiplicative gain curves.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct CompressorX2 {
    pub k: [CompressorKnee; 2],
}

/// Gate knee model with three regions:
/// 1. Below `start`: constant `gain_start`
/// 2. Between `start` and `end`: cubic Hermite transition
/// 3. Above `end`: constant `gain_end`
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct GateKnee {
    /// Transition start threshold (linear).
    pub start: f32,
    /// Transition end threshold (linear).
    pub end: f32,
    /// Gain below start threshold.
    pub gain_start: f32,
    /// Gain above end threshold.
    pub gain_end: f32,
    /// Hermite interpolation coefficients (cubic in log domain).
    pub herm: [f32; 4],
}

/// Expander knee model supporting upward/downward expansion.
///
/// Uses threshold limiting and soft knee transitions with Hermite
/// polynomial interpolation.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct ExpanderKnee {
    /// Knee start point.
    pub start: f32,
    /// Knee end point.
    pub end: f32,
    /// Hard threshold for limiting.
    pub threshold: f32,
    /// Hermite interpolation coefficients (quadratic in log domain).
    pub herm: [f32; 3],
    /// Linear expansion parameters: `tilt[0]*ln(x) + tilt[1]`.
    pub tilt: [f32; 2],
}

// ─── DSP Context ───────────────────────────────────────────────────────────

/// DSP processing context for storing/restoring machine state.
///
/// Manages floating-point control registers to enable flush-to-zero (FTZ)
/// and denormals-are-zero (DAZ) modes during audio processing. Without
/// these modes, denormalized floats in recursive filters can cause
/// 10-100x slowdowns.
///
/// # Platform support
///
/// - **x86/x86_64**: Sets FTZ and DAZ bits in the MXCSR register.
/// - **aarch64**: Sets the FZ bit in the FPCR register.
/// - **Other**: No-op (denormal handling is not available).
///
/// # Examples
/// ```
/// use lsp_dsp_lib::types::DspContext;
///
/// let mut ctx = DspContext::default();
/// ctx.start();
/// // ... DSP processing with denormals flushed to zero ...
/// ctx.finish();
/// ```
#[derive(Debug, Clone, Default)]
pub struct DspContext {
    /// Saved floating-point control register value, restored on `finish()`.
    saved_fpcr: u64,
    /// Whether this context has been started (and not yet finished).
    active: bool,
}

/// FTZ bit in x86 MXCSR register.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const MXCSR_FTZ: u32 = 0x8000;

/// DAZ bit in x86 MXCSR register.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const MXCSR_DAZ: u32 = 0x0040;

impl DspContext {
    /// Begin DSP processing. Enables flush-to-zero mode for denormals.
    ///
    /// The previous floating-point control state is saved and will be
    /// restored when [`finish`](Self::finish) is called.
    pub fn start(&mut self) {
        if self.active {
            return;
        }
        self.active = true;
        self.enable_ftz();
    }

    /// End DSP processing. Restores the floating-point control state
    /// that was saved by [`start`](Self::start).
    pub fn finish(&mut self) {
        if !self.active {
            return;
        }
        self.active = false;
        self.restore_fpcr();
    }

    /// Returns `true` if this context is currently active (between
    /// `start()` and `finish()` calls).
    pub fn is_active(&self) -> bool {
        self.active
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn enable_ftz(&mut self) {
        unsafe {
            #[cfg(target_arch = "x86")]
            use core::arch::x86::{_mm_getcsr, _mm_setcsr};
            #[cfg(target_arch = "x86_64")]
            use core::arch::x86_64::{_mm_getcsr, _mm_setcsr};

            let csr = _mm_getcsr();
            self.saved_fpcr = csr as u64;
            _mm_setcsr(csr | MXCSR_FTZ | MXCSR_DAZ);
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn restore_fpcr(&self) {
        unsafe {
            #[cfg(target_arch = "x86")]
            use core::arch::x86::_mm_setcsr;
            #[cfg(target_arch = "x86_64")]
            use core::arch::x86_64::_mm_setcsr;

            _mm_setcsr(self.saved_fpcr as u32);
        }
    }

    #[cfg(target_arch = "aarch64")]
    fn enable_ftz(&mut self) {
        unsafe {
            let fpcr: u64;
            core::arch::asm!("mrs {}, fpcr", out(reg) fpcr);
            self.saved_fpcr = fpcr;
            // Set FZ bit (bit 24)
            core::arch::asm!("msr fpcr, {}", in(reg) fpcr | (1u64 << 24));
        }
    }

    #[cfg(target_arch = "aarch64")]
    fn restore_fpcr(&self) {
        unsafe {
            core::arch::asm!("msr fpcr, {}", in(reg) self.saved_fpcr);
        }
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
    fn enable_ftz(&mut self) {
        // No-op on unsupported platforms
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
    fn restore_fpcr(&self) {
        // No-op on unsupported platforms
    }
}

impl Drop for DspContext {
    fn drop(&mut self) {
        if self.active {
            self.restore_fpcr();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_biquad_default() {
        let bq = Biquad::default();
        assert!(bq.d.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_biquad_reset() {
        let mut bq = Biquad::default();
        bq.d[0] = 1.0;
        bq.d[5] = -3.0;
        bq.reset();
        assert!(bq.d.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_compressor_knee_default() {
        let knee = CompressorKnee::default();
        assert_eq!(knee.start, 0.0);
        assert_eq!(knee.gain, 0.0);
        assert_eq!(knee.herm, [0.0; 3]);
        assert_eq!(knee.tilt, [0.0; 2]);
    }

    #[test]
    fn test_gate_knee_default() {
        let knee = GateKnee::default();
        assert_eq!(knee.herm, [0.0; 4]);
    }

    #[test]
    fn test_filter_cascade_default() {
        let fc = FilterCascade::default();
        assert_eq!(fc.t, [0.0; 4]);
        assert_eq!(fc.b, [0.0; 4]);
    }

    #[test]
    fn test_dsp_context_start_finish() {
        let mut ctx = DspContext::default();
        assert!(!ctx.is_active());

        ctx.start();
        assert!(ctx.is_active());

        ctx.finish();
        assert!(!ctx.is_active());
    }

    #[test]
    fn test_dsp_context_double_start() {
        let mut ctx = DspContext::default();
        ctx.start();
        ctx.start(); // should be idempotent
        assert!(ctx.is_active());
        ctx.finish();
        assert!(!ctx.is_active());
    }

    #[test]
    fn test_dsp_context_double_finish() {
        let mut ctx = DspContext::default();
        ctx.start();
        ctx.finish();
        ctx.finish(); // should be idempotent
        assert!(!ctx.is_active());
    }

    #[test]
    fn test_dsp_context_denormals_flushed() {
        // The smallest positive normal f32 is ~1.18e-38.
        // A denormal f32 is smaller than that.
        let denormal: f32 = f32::MIN_POSITIVE * 0.5;
        assert!(
            denormal != 0.0,
            "precondition: denormal should be non-zero without FTZ"
        );
        assert!(
            denormal.is_subnormal(),
            "precondition: value should be subnormal"
        );

        let mut ctx = DspContext::default();
        ctx.start();

        // With FTZ/DAZ enabled, operations on denormals should produce zero.
        // Adding a denormal to zero and storing through a volatile prevents
        // the compiler from optimizing it away.
        let result: f32;
        unsafe {
            let mut tmp = 0.0f32;
            let ptr = &mut tmp as *mut f32;
            core::ptr::write_volatile(ptr, denormal + 0.0);
            result = core::ptr::read_volatile(ptr);
        }

        // On platforms that support FTZ, the result should be zero.
        // On unsupported platforms, it remains denormal.
        #[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
        assert_eq!(
            result, 0.0,
            "denormal should be flushed to zero with FTZ enabled"
        );

        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
        let _ = result; // no assertion on unsupported platforms

        ctx.finish();

        // After restoring, denormals should be preserved again
        let check: f32;
        unsafe {
            let mut tmp = 0.0f32;
            let ptr = &mut tmp as *mut f32;
            core::ptr::write_volatile(ptr, denormal + 0.0);
            check = core::ptr::read_volatile(ptr);
        }
        assert_ne!(check, 0.0, "denormals should be preserved after finish()");
    }

    #[test]
    fn test_dsp_context_drop_restores_state() {
        let denormal: f32 = f32::MIN_POSITIVE * 0.5;
        assert!(denormal.is_subnormal());

        {
            let mut ctx = DspContext::default();
            ctx.start();
            // ctx is dropped here without calling finish()
        }

        // After drop, FTZ should be restored to the previous state
        let check: f32;
        unsafe {
            let mut tmp = 0.0f32;
            let ptr = &mut tmp as *mut f32;
            core::ptr::write_volatile(ptr, denormal + 0.0);
            check = core::ptr::read_volatile(ptr);
        }
        assert_ne!(
            check, 0.0,
            "denormals should be preserved after DspContext is dropped"
        );
    }
}
