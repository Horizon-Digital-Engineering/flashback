#![no_main]

use libfuzzer_sys::fuzz_target;

// The one function here that slices by byte offset. It returns a borrowed
// window into its input, so the window has to BE a window — a slice taken on
// the wrong side of a character boundary panics, and one that invents content
// would mean the parser is reading something the model never sent.
fuzz_target!(|data: &[u8]| {
    let Ok(text) = std::str::from_utf8(data) else {
        return;
    };
    let out = flashback_nlp::provider::remote::strip_fences(text);
    assert!(
        text.contains(out),
        "strip_fences returned {out:?}, which is not part of its input"
    );
    assert!(out.len() <= text.len());
});
