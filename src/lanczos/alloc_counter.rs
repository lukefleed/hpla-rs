//! Counting global allocator for the zero-allocation regression test.
//!
//! Compiled only under `cfg(test)`. Wraps [`std::alloc::System`] and bumps
//! two relaxed atomic counters on every `alloc`, `alloc_zeroed`, and
//! `realloc`. Tests call [`reset`] before the region of interest and
//! [`snapshot`] after to obtain the bytes-and-count delta.
//!
//! The counters are process-global and see every allocation on every
//! thread, so a test that runs concurrently with another allocating test
//! may observe noise. Invariance assertions (delta between two problem
//! sizes) are more robust than absolute thresholds under parallelism.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};

static TOTAL_BYTES: AtomicU64 = AtomicU64::new(0);
static TOTAL_COUNT: AtomicU64 = AtomicU64::new(0);

/// Counting wrapper around the platform [`System`] allocator.
pub(crate) struct CountingAllocator;

// SAFETY: forwards every call to `std::alloc::System`, which is a valid
// `GlobalAlloc` implementation. The counter bumps use relaxed atomics and
// do not participate in the memory-safety contract.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        TOTAL_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        TOTAL_COUNT.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        TOTAL_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        TOTAL_COUNT.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        TOTAL_BYTES.fetch_add(new_size as u64, Ordering::Relaxed);
        TOTAL_COUNT.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

/// Zeroes the global counters. Call before the region of interest.
pub(crate) fn reset() {
    TOTAL_BYTES.store(0, Ordering::Relaxed);
    TOTAL_COUNT.store(0, Ordering::Relaxed);
}

/// Returns `(total_bytes, total_count)` accumulated since the last
/// [`reset`]. Call after the region of interest.
pub(crate) fn snapshot() -> (u64, u64) {
    (
        TOTAL_BYTES.load(Ordering::Relaxed),
        TOTAL_COUNT.load(Ordering::Relaxed),
    )
}
