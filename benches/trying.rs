use trying::trie::{Trie, TrieAtom, TrieKey, TrieString, TrieValue, TrieVec};

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};
use rand::{
    distr::{Alphanumeric, Uniform},
    rng, Rng,
};

fn get_text() -> Vec<String> {
    use std::fs::File;
    use std::io::Read;
    const DATA: &[&str] = &["data/1984.txt", "data/sun-rising.txt"];
    let mut contents = String::new();
    File::open(DATA[1])
        .unwrap()
        .read_to_string(&mut contents)
        .unwrap();
    contents
        .split(|c: char| c.is_whitespace())
        .map(|s| s.to_string())
        .collect()
}

fn make_trie(words: &[String]) -> TrieString<usize> {
    let mut trie = Trie::new();
    for w in words {
        let len = w.len();
        trie.insert_with_value(w.chars(), Some(len));
    }
    trie
}

fn trie_insert(b: &mut Criterion) {
    let words = get_text();
    b.bench_function("trie insert", |b| b.iter(|| make_trie(&words)));
}

fn trie_get(b: &mut Criterion) {
    let words = get_text();
    let trie = make_trie(&words);
    b.bench_function("trie get", |b| {
        b.iter(|| {
            words
                .iter()
                .map(|w| trie.get(w.chars()))
                .collect::<Vec<Option<&usize>>>()
        })
    });
}

fn trie_insert_remove(b: &mut Criterion) {
    let words = get_text();

    b.bench_function("trie remove", |b| {
        b.iter(|| {
            let mut trie = make_trie(&words);
            for w in &words {
                trie.remove(w.chars());
            }
        });
    });
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut trie = TrieString::<usize>::new();
    c.bench_function("inserting: char items (len: 1..=512)", |b| {
        b.iter_batched(
            || {
                rng()
                    .sample_iter(&Alphanumeric)
                    .take(rng().random_range(1..=512))
                    .map(char::from)
            },
            |input| insert_trie(&mut trie, input),
            BatchSize::SmallInput,
        )
    });
    c.bench_function("contains: char items (len: 1..=512)", |b| {
        b.iter_batched(
            || {
                rng()
                    .sample_iter(&Alphanumeric)
                    .take(rng().random_range(1..=512))
                    .map(char::from)
            },
            |input| contains_trie(&trie, input),
            BatchSize::SmallInput,
        )
    });
    trie.clear();
}

fn iterate(c: &mut Criterion) {
    static BASE_SIZE: usize = 16;
    static POPULATION_SIZE: usize = 1000;

    let mut group = c.benchmark_group("iterate");
    for size in [
        BASE_SIZE,
        2 * BASE_SIZE,
        4 * BASE_SIZE,
        8 * BASE_SIZE,
        16 * BASE_SIZE,
        32 * BASE_SIZE,
        64 * BASE_SIZE,
    ]
    .iter()
    {
        group.throughput(Throughput::Bytes(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("consuming iteration (char)", size),
            size,
            |b, &size| {
                let mut trie = TrieString::<usize>::new();
                for _i in 0..POPULATION_SIZE {
                    let entry: Vec<char> = rng()
                        .sample_iter(&Alphanumeric)
                        .take(rng().random_range(1..=size))
                        .map(char::from)
                        .collect();
                    trie.insert(entry);
                }
                b.iter_batched(|| trie.clone(), iterate_trie, BatchSize::SmallInput)
            },
        );
        group.bench_with_input(
            BenchmarkId::new("reference iteration (char)", size),
            size,
            |b, &size| {
                let mut trie = TrieString::<usize>::new();
                for _i in 0..POPULATION_SIZE {
                    let entry: Vec<char> = rng()
                        .sample_iter(&Alphanumeric)
                        .take(rng().random_range(1..=size))
                        .map(char::from)
                        .collect();
                    trie.insert(entry);
                }
                b.iter_batched(|| {}, |_| iterate_trie_ref(&trie), BatchSize::SmallInput)
            },
        );
    }
    group.finish();
}

fn search(c: &mut Criterion) {
    static BASE_SIZE: usize = 16;
    static POPULATION_SIZE: usize = 10000;

    let mut group = c.benchmark_group("search");
    for size in [
        BASE_SIZE,
        2 * BASE_SIZE,
        4 * BASE_SIZE,
        8 * BASE_SIZE,
        16 * BASE_SIZE,
        32 * BASE_SIZE,
        64 * BASE_SIZE,
    ]
    .iter()
    {
        let range = Uniform::new_inclusive(1, size).unwrap();
        group.throughput(Throughput::Bytes(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("random find (usize)", size),
            size,
            |b, &size| {
                let mut trie = TrieVec::<usize, usize>::new();
                for _i in 0..POPULATION_SIZE {
                    let entry: Vec<usize> = rng()
                        .sample_iter(range)
                        .take(rng().random_range(1..=size))
                        .collect();
                    trie.insert(entry);
                }
                b.iter_batched(
                    || rng().sample_iter(range).take(rng().random_range(1..=size)),
                    |input| contains_trie(&trie, input),
                    BatchSize::SmallInput,
                )
            },
        );
        group.bench_with_input(
            BenchmarkId::new("always find (usize)", size),
            size,
            |b, &size| {
                let mut trie = TrieVec::<usize, usize>::new();
                let mut searches: Vec<Vec<usize>> = vec![];
                for _i in 0..POPULATION_SIZE {
                    let entry: Vec<usize> = rng()
                        .sample_iter(range)
                        .take(rng().random_range(1..=size))
                        .collect();
                    searches.push(entry.clone());
                    trie.insert(entry);
                }
                b.iter_batched(
                    || searches[rng().random_range(1..POPULATION_SIZE)].clone(),
                    |input| contains_trie(&trie, input),
                    BatchSize::SmallInput,
                )
            },
        );
        group.bench_with_input(
            BenchmarkId::new("random find (char)", size),
            size,
            |b, &size| {
                let mut trie = TrieString::<usize>::new();
                for _i in 0..POPULATION_SIZE {
                    let entry: Vec<char> = rng()
                        .sample_iter(&Alphanumeric)
                        .take(rng().random_range(1..=size))
                        .map(char::from)
                        .collect();
                    trie.insert(entry);
                }
                b.iter_batched(
                    || {
                        rng()
                            .sample_iter(&Alphanumeric)
                            .take(rng().random_range(1..=size))
                            .map(char::from)
                    },
                    |input| contains_trie(&trie, input),
                    BatchSize::SmallInput,
                )
            },
        );
        group.bench_with_input(
            BenchmarkId::new("always find (char)", size),
            size,
            |b, &size| {
                let mut trie = TrieString::<usize>::new();
                let mut searches: Vec<Vec<char>> = vec![];
                for _i in 0..POPULATION_SIZE {
                    let entry: Vec<char> = rng()
                        .sample_iter(&Alphanumeric)
                        .take(rng().random_range(1..=size))
                        .map(char::from)
                        .collect();
                    searches.push(entry.clone());
                    trie.insert(entry);
                }
                b.iter_batched(
                    || searches[rng().random_range(1..POPULATION_SIZE)].clone(),
                    |input| contains_trie(&trie, input),
                    BatchSize::SmallInput,
                )
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    trie_insert,
    trie_get,
    trie_insert_remove,
    criterion_benchmark,
    search,
    iterate
);
criterion_main!(benches);

fn insert_trie<S: IntoIterator<Item = A>, K: TrieKey<A>, A: TrieAtom, V: TrieValue>(
    trie: &mut Trie<K, A, V>,
    input: S,
) {
    trie.insert(input);
}

fn contains_trie<S: IntoIterator<Item = A>, K: TrieKey<A>, A: TrieAtom, V: TrieValue>(
    trie: &Trie<K, A, V>,
    input: S,
) {
    trie.contains(input);
}

fn iterate_trie<K: TrieKey<A>, A: TrieAtom, V: TrieValue>(trie: Trie<K, A, V>) {
    trie.into_iter().for_each(|_x| ());
}

fn iterate_trie_ref<K: TrieKey<A>, A: TrieAtom, V: TrieValue>(trie: &Trie<K, A, V>) {
    trie.iter().for_each(|_x| ());
}
