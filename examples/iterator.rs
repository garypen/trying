use rand::{distr::Alphanumeric, rng, Rng};
use std::iter::FromIterator;
use trying::trie::Trie;

fn main() {
    static POPULATION_SIZE: usize = 10;
    static SIZE: usize = 10;

    // Create our trie and a collection of searches
    let mut trie = Trie::new();
    let mut searches = vec![];

    // Store 10 random strings (char sequences)
    // composed of between 1 and 10 characters in
    // our search collection and our trie.
    for _i in 0..POPULATION_SIZE {
        let entry: Vec<char> = rng()
            .sample_iter(&Alphanumeric)
            .take(rng().random_range(1..=SIZE))
            .map(char::from)
            .collect();
        searches.push(entry.clone());
        let len = entry.len();
        trie.insert_with_value(entry, Some(len));
    }

    // iterate over our search collection and confirm
    // that all keys are in our search collection
    println!("unsorted");
    for pair in trie.iter() {
        assert!(searches.contains(&pair.key));
        println!(
            "key: {}, value: {:?}",
            String::from_iter(&pair.key),
            pair.value
        );
    }
    println!("sorted");
    for pair in trie.iter_sorted() {
        assert!(searches.contains(&pair.key));
        println!(
            "key: {}, value: {:?}",
            String::from_iter(&pair.key),
            pair.value
        );
    }
}
