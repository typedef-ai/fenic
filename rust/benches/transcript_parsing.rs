use criterion::{black_box, criterion_group, criterion_main, Criterion};
use polars_plugins::transcript::ParserRegistry;

fn bench_transcript_parsers(c: &mut Criterion) {
    let transcript_text = include_str!("../data/transcripts/transcript-1.txt");
    let transcript_text_srt = include_str!("../data/transcripts/transcript-1.srt");
    let transcript_text_webvtt = include_str!("../data/transcripts/transcript-1.webvtt");
    let mut group = c.benchmark_group("transcript parsers");

    let registry = ParserRegistry::default();

    group.bench_function("Generic parser", |b| {
        b.iter(|| {
            let res = registry
                .parse("generic", black_box(transcript_text))
                .unwrap();
            black_box(res);
        });
    });

    group.bench_function("SRT parser", |b| {
        b.iter(|| {
            let res = registry
                .parse("srt", black_box(transcript_text_srt))
                .unwrap();
            black_box(res);
        });
    });

    group.bench_function("WebVTT parser", |b| {
        b.iter(|| {
            let res = registry
                .parse("webvtt", black_box(transcript_text_webvtt))
                .unwrap();
            black_box(res);
        });
    });

    group.finish();
}

criterion_group!(benches, bench_transcript_parsers);
criterion_main!(benches);
