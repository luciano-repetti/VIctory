//! One-time patches of a mapped image's bytes before the title runs
//! (`VICTORY_PATCH`), read-only sections included.

use windows_sys::Win32::System::Memory::{PAGE_READWRITE, VirtualProtect};

/// Writes `bytes` at `base + rva` of an image of `size` bytes, lifting the
/// page protection for the write and restoring it after. Call before the
/// title runs: nothing else may read the bytes meanwhile.
pub fn patch_image(base: usize, size: usize, rva: usize, bytes: &[u8]) -> Result<(), String> {
    let end = rva
        .checked_add(bytes.len())
        .filter(|&end| end <= size && !bytes.is_empty())
        .ok_or_else(|| format!("{rva:#x}+{}: outside the image", bytes.len()))?;
    let at = base + rva;
    let mut old = 0;
    // SAFETY: a range inside the mapped image (checked above).
    if unsafe { VirtualProtect(at as _, end - rva, PAGE_READWRITE, &mut old) } == 0 {
        return Err(format!("{rva:#x}: {}", std::io::Error::last_os_error()));
    }
    // SAFETY: now writable, `bytes.len()` bytes inside the image.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), at as *mut u8, bytes.len()) };
    let mut ignored = 0;
    // SAFETY: as above; the protection the range had.
    unsafe { VirtualProtect(at as _, end - rva, old, &mut ignored) };
    Ok(())
}

/// The patches of a `VICTORY_PATCH` value: `<rva>:<hex bytes>,...`, the
/// rva hex (with or without 0x), the bytes as pairs of hex digits.
pub fn parse_patches(value: &str) -> Result<Vec<(usize, Vec<u8>)>, String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .map(|p| {
            let (rva, hex) = p.split_once(':').ok_or_else(|| format!("{p}: expected <rva>:<hex bytes>"))?;
            let rva = usize::from_str_radix(rva.trim().trim_start_matches("0x"), 16)
                .map_err(|e| format!("{p}: rva: {e}"))?;
            let hex: String = hex.chars().filter(|c| !c.is_whitespace()).collect();
            if hex.is_empty() || !hex.len().is_multiple_of(2) {
                return Err(format!("{p}: bytes must be pairs of hex digits"));
            }
            let bytes = (0..hex.len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&hex[i..i + 2], 16))
                .collect::<Result<Vec<u8>, _>>()
                .map_err(|e| format!("{p}: bytes: {e}"))?;
            Ok((rva, bytes))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patches_parse_or_say_why_not() {
        assert_eq!(
            parse_patches("615c100:8007000038040000, 0x10:ff").unwrap(),
            vec![(0x615c100, vec![0x80, 7, 0, 0, 0x38, 4, 0, 0]), (0x10, vec![0xff])]
        );
        for bad in ["615c100", "615c100:", "615c100:abc", "zz:00", "10:0g"] {
            assert!(parse_patches(bad).is_err(), "{bad}");
        }
        assert!(parse_patches("").unwrap().is_empty());
    }

    #[test]
    fn a_read_only_page_is_patched_and_stays_read_only() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READONLY, VirtualAlloc, VirtualFree,
        };
        // SAFETY: a fresh page, made read-only, patched, freed.
        unsafe {
            let page = VirtualAlloc(std::ptr::null(), 4096, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
            assert!(!page.is_null());
            let mut old = 0;
            assert_ne!(VirtualProtect(page, 4096, PAGE_READONLY, &mut old), 0);
            patch_image(page as usize, 4096, 0x100, &[1, 2, 3]).unwrap();
            assert_eq!(std::slice::from_raw_parts((page as *const u8).add(0x100), 3), [1, 2, 3]);
            assert_ne!(VirtualProtect(page, 4096, PAGE_READONLY, &mut old), 0);
            assert_eq!(old, PAGE_READONLY, "the protection is back");
            assert!(patch_image(page as usize, 4096, 4095, &[1, 2]).is_err(), "past the end");
            VirtualFree(page, 0, MEM_RELEASE);
        }
    }
}
