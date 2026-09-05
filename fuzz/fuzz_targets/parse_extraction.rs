#![no_main]

use libfuzzer_sys::fuzz_target;

// Every byte here reaches us as the body of a model's reply. The provider is
// told to answer in JSON and frequently doesn't: prose, fences, a truncated
// object, a field of the wrong type. None of that may panic, and a reply the
// parser accepts has to survive being re-read.
fuzz_target!(|data: &[u8]| {
    let Ok(text) = std::str::from_utf8(data) else {
        return;
    };
    if let Ok(extraction) = flashback_nlp::provider::remote::parse_extraction(text) {
        let round = serde_json::to_string(&extraction).expect("an Extraction always serializes");
        let back: flashback_nlp::Extraction =
            serde_json::from_str(&round).expect("our own output must parse");
        assert_eq!(back.intent, extraction.intent);
        assert_eq!(back.entities.len(), extraction.entities.len());
    }
});
