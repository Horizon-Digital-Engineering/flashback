#![no_main]

use libfuzzer_sys::fuzz_target;

// The distillation reply becomes semantic facts, which are what retrieval
// serves first. A malformed batch must fail as BadOutput, never take the
// curation pass down with it.
fuzz_target!(|data: &[u8]| {
    let Ok(text) = std::str::from_utf8(data) else {
        return;
    };
    if let Ok(facts) = flashback_nlp::provider::remote::parse_distill(text) {
        for f in &facts {
            assert!(f.confidence.is_finite(), "a fact scored {}", f.confidence);
        }
    }
});
