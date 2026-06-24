//! Load a `.model` and print a summary — proves the loader stack end-to-end.
//!
//! Usage: cargo run --example inspect -- tests/models/botchan_1000_bpe.model

use sentencepiece_rs::SentencePieceProcessor;

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "tests/models/botchan_1000_bpe.model".to_string());

    let sp = match SentencePieceProcessor::open(&path) {
        Ok(sp) => sp,
        Err(e) => {
            eprintln!("failed to load {path}: {e}");
            std::process::exit(1);
        }
    };

    println!("model:       {path}");
    println!("model_type:  {:?}", sp.model_type());
    println!("piece_size:  {}", sp.piece_size());
    println!("first pieces:");
    for id in 0..20.min(sp.piece_size() as i32) {
        if let Some(p) = sp.id_to_piece(id) {
            println!("  {id:>4}  {p:?}");
        }
    }

    let sample = "the quick brown fox";
    match sp.encode(sample) {
        Ok(ids) => {
            println!("encode({sample:?}) = {ids:?}");
            let pieces: Vec<&str> = ids.iter().filter_map(|&i| sp.id_to_piece(i)).collect();
            println!("pieces = {pieces:?}");
            if let Ok(text) = sp.decode(&ids) {
                println!("decode = {text:?}");
            }
        }
        Err(e) => println!("encode({sample:?}) failed: {e}"),
    }
}
