//! Rough throughput benchmark for encode/decode. Zero dependencies — just
//! `std::time`. Build with `--release` or the numbers are meaningless.
//!
//! Usage:
//!   cargo run --release --example bench -- <model> <corpus> [iters]
//!
//! Pair it with `oracle/bench_oracle.py` (same args) to compare against the
//! Python/C++ reference. Not apples-to-apples — this is single-threaded
//! straightforward code optimised for matching the oracle, not for speed — but
//! it shows the port works at a sane throughput.

use std::time::Instant;

use sentencepiece_rs::SentencePieceProcessor;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "tests/models/botchan_1000_bpe.model".to_string());
    let corpus = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "oracle/corpus_ascii.txt".to_string());
    let iters: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(50);

    let sp = SentencePieceProcessor::open(&model).expect("load model");
    let text = std::fs::read_to_string(&corpus).expect("read corpus");
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    let total_bytes: usize = lines.iter().map(|l| l.len()).sum();

    // Warm up (allocator, branch predictor, file cache).
    for l in &lines {
        let _ = sp.encode(l);
    }

    // Encode.
    let mut tokens = 0usize;
    let t = Instant::now();
    for _ in 0..iters {
        for l in &lines {
            tokens += sp.encode(l).unwrap().len();
        }
    }
    let enc = t.elapsed().as_secs_f64();

    // Decode (over the encoded ids).
    let encoded: Vec<Vec<i32>> = lines.iter().map(|l| sp.encode(l).unwrap()).collect();
    let t = Instant::now();
    for _ in 0..iters {
        for ids in &encoded {
            let _ = sp.decode(ids).unwrap();
        }
    }
    let dec = t.elapsed().as_secs_f64();

    let n = (lines.len() * iters) as f64;
    let mb = (total_bytes * iters) as f64 / 1e6;
    println!("model:  {model}");
    println!(
        "corpus: {corpus} ({} lines, {} bytes) x{iters} iters; {tokens} tokens",
        lines.len(),
        total_bytes
    );
    println!(
        "encode: {enc:.3}s  {:.0} lines/s  {:.1} MB/s",
        n / enc,
        mb / enc
    );
    println!(
        "decode: {dec:.3}s  {:.0} lines/s  {:.1} MB/s",
        n / dec,
        mb / dec
    );
}
