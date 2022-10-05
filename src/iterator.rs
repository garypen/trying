//! Provides Trie iterators.
//!
//! Here's an example of how we can iterate over our Trie. We use the
//! `FromIterator` trait to reconstruct our source key from the
//! vector of atoms which the iterator returns as the key.
//!
//! NB: Because we stripped all of the whitespace out when we split our
//! key, we need to re-add it before we insert it. Until `intersperse`
//! is added to the std library, the simplest way to do this right now
//! is to use itertools.
//!
//! Example 4
//! ```
//! use std::iter::FromIterator;
//! use itertools::Itertools;
//! use trying::trie::TrieVec;
//!
//! let mut trie = TrieVec::<&str, usize>::new();
//! let input = Itertools::intersperse("the quick brown fox".split_whitespace(), " ");
//! trie.insert_with_value(input.clone(), Some(4));
//!
//! // Anything which implements IntoIterator<Item=&str> can now be used
//! // to interact with our Trie
//! for kv_pair in trie.into_iter() {
//!     println!("kv_pair: {:?}", kv_pair);
//!     assert_eq!("the quick brown fox", String::from_iter(kv_pair.key));
//!     assert_eq!(kv_pair.value, Some(4));
//! }
//! ```
//!

use crate::trie::{Node, Trie, TrieAtom, TrieKey, TrieValue};

/// Iterator Item
#[derive(Clone, Debug)]
pub struct KeyValue<K, A, V> {
    pub key: K,
    pub value: Option<V>,
    phantom: std::marker::PhantomData<A>,
}

/// Iterator Item
#[derive(Clone, Debug)]
pub struct KeyValueRef<'a, K, A, V> {
    pub key: K,
    pub value: Option<&'a V>,
    phantom: std::marker::PhantomData<A>,
}

/// Iterator over a Trie.
#[derive(Debug)]
pub struct TrieIntoIterator<K, A, V> {
    results: Vec<KeyValue<K, A, V>>,
    backtrack: usize,
    nodes: Vec<Node<A, V>>,
}

impl<K: TrieKey<A>, A: TrieAtom, V: TrieValue> IntoIterator for Trie<K, A, V> {
    type Item = KeyValue<K, A, V>;
    type IntoIter = TrieIntoIterator<K, A, V>;

    fn into_iter(self) -> Self::IntoIter {
        let mut results: Vec<Self::Item> = vec![];

        let mut nodes = vec![self.head];

        // Create our seed column and results
        Trie::<K, A, V>::make_column(&mut nodes);
        Trie::create_results(0, &mut results, &mut nodes[1..]);

        results.reverse();
        TrieIntoIterator {
            results,
            backtrack: 0,
            nodes,
        }
    }
}

impl<K: TrieKey<A>, A: TrieAtom, V: TrieValue> Iterator for TrieIntoIterator<K, A, V> {
    type Item = KeyValue<K, A, V>;

    fn next(&mut self) -> Option<Self::Item> {
        // Keep return results from our current column, when that is empty try
        // to create a new column.
        match self.results.pop() {
            Some(v) => Some(v),
            None => {
                // Create a new column from results until there are no more columns.
                if self.nodes.len() > 1 {
                    let finish = self.nodes.len() - 1;
                    for (idx, node) in self.nodes.iter().rev().enumerate() {
                        if !node.children.is_empty() || idx == finish {
                            self.backtrack = self.nodes.len() - idx;
                            break;
                        }
                    }
                    self.nodes.truncate(self.backtrack);
                    Trie::<K, A, V>::make_column(&mut self.nodes);
                    Trie::create_results(self.backtrack, &mut self.results, &mut self.nodes[1..]);
                }
                self.results.reverse();
                self.results.pop()
            }
        }
    }
}

#[derive(Debug)]
struct NodeRef<'a, A: TrieAtom, V: TrieValue>(&'a Node<A, V>, usize);

/// Iterator over a Trie.
#[derive(Debug)]
pub struct TrieRefIntoIterator<'a, K: TrieKey<A>, A: TrieAtom, V: TrieValue> {
    results: Vec<KeyValueRef<'a, K, A, V>>,
    backtrack: usize,
    nodes: Vec<NodeRef<'a, A, V>>,
}

// Iterator
impl<'a, A: TrieAtom, V: TrieValue, K: TrieKey<A>> IntoIterator for &'a Trie<K, A, V> {
    type Item = KeyValueRef<'a, K, A, V>;
    type IntoIter = TrieRefIntoIterator<'a, K, A, V>;

    fn into_iter(self) -> Self::IntoIter {
        let mut results: Vec<Self::Item> = vec![];

        let mut nodes = vec![NodeRef(&self.head, Default::default())];

        // Create our seed column and results
        Trie::<K, A, V>::make_tracked_column(&mut nodes);
        Trie::create_tracked_results(0, &mut results, &nodes[1..]);

        results.reverse();
        TrieRefIntoIterator {
            results,
            backtrack: 0,
            nodes,
        }
    }
}

impl<'a, A: TrieAtom, V: TrieValue, K: TrieKey<A>> Iterator for TrieRefIntoIterator<'a, K, A, V> {
    type Item = KeyValueRef<'a, K, A, V>;

    fn next(&mut self) -> Option<Self::Item> {
        // Keep return results from our current column, when that is empty try
        // to create a new column.
        match self.results.pop() {
            Some(v) => Some(v),
            None => {
                // Create a new column from results until there are no more columns.
                if self.nodes.len() > 1 {
                    let finish = self.nodes.len() - 1;
                    for (idx, node) in self.nodes.iter().rev().enumerate() {
                        if node.0.children.len() > node.1 || idx == finish {
                            self.backtrack = self.nodes.len() - idx;
                            break;
                        }
                    }
                    self.nodes.truncate(self.backtrack);
                    Trie::<K, A, V>::make_tracked_column(&mut self.nodes);
                    Trie::create_tracked_results(
                        self.backtrack,
                        &mut self.results,
                        &self.nodes[1..],
                    );
                }
                self.results.reverse();
                self.results.pop()
            }
        }
    }
}

// Useful utility functions for building iterator output
impl<'a, A: TrieAtom, V: TrieValue, K: TrieKey<A>> Trie<K, A, V> {
    #[inline(always)]
    fn make_column(nodes: &mut Vec<Node<A, V>>) {
        loop {
            let index = nodes.len() - 1;
            let node = match nodes.get_mut(index) {
                Some(n) => n,
                None => break,
            };
            if !node.children.is_empty() {
                let child = node.children.remove(0);
                nodes.push(child);
            } else {
                break;
            }
        }
    }

    #[inline(always)]
    fn make_tracked_column(nodes: &mut Vec<NodeRef<A, V>>) {
        loop {
            let index = nodes.len() - 1;
            let mut node = match nodes.get_mut(index) {
                Some(n) => n,
                None => break,
            };
            if node.0.children.len() > node.1 {
                let child = node.0.children.get(node.1).unwrap();
                node.1 += 1;
                nodes.push(NodeRef(child, Default::default()));
            } else {
                break;
            }
        }
    }

    #[inline(always)]
    fn create_results(
        backtrack: usize,
        results: &mut Vec<KeyValue<K, A, V>>,
        nodes: &mut [Node<A, V>],
    ) {
        let mut current = vec![];
        for node in nodes {
            current.push(node.pair.atom);
            if current.len() >= backtrack && node.terminated {
                results.push(KeyValue {
                    key: K::from_iter(current.clone()),
                    value: node.pair.value.take(),
                    phantom: Default::default(),
                });
            }
        }
    }

    #[inline(always)]
    fn create_tracked_results<'b: 'a>(
        backtrack: usize,
        results: &mut Vec<KeyValueRef<'b, K, A, V>>,
        nodes: &'a [NodeRef<'b, A, V>],
    ) {
        let mut current = vec![];
        for node in nodes {
            current.push(node.0.pair.atom);
            if current.len() >= backtrack && node.0.terminated {
                results.push(KeyValueRef {
                    key: K::from_iter(current.clone()),
                    value: node.0.pair.value.as_ref(),
                    phantom: Default::default(),
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::trie::{TrieString, TrieVec};
    use itertools::Itertools;
    use rand::{distributions::Alphanumeric, thread_rng, Rng};
    use std::collections::HashSet;

    #[test]
    fn it_iterates_over_empty_trie() {
        let trie = TrieString::<usize>::new();
        trie.iter().for_each(|_x| ());
        assert_eq!(0, trie.count());
    }

    #[test]
    fn it_iterates_and_re_assembles_trie() {
        let mut trie = TrieVec::<&str, usize>::new();
        let input = "the quick brown fox".split_whitespace();
        trie.insert_with_value(input.clone(), Some(4));
        let input = "the quick brown cat".split_whitespace();
        trie.insert_with_value(input.clone(), Some(4));
        let input = "lazy dog".split_whitespace();
        trie.insert_with_value(input.clone(), Some(2));

        for kv_pair in trie.iter_sorted() {
            println!("kv_pair: {:?}", kv_pair);
        }

        if let Some(kv_pair) = trie.into_iter().next() {
            println!("kv_pair: {:?}", kv_pair);
            assert_eq!(
                "the quick brown fox",
                Itertools::intersperse(kv_pair.key.into_iter(), " ").collect::<String>()
            );
        } else {
            panic!("did not get first line from iterator");
        }
    }

    #[test]
    fn it_iterates_over_owned_populated_trie() {
        let mut trie = TrieString::<usize>::new();
        let mut input = vec!["abcdef", "abcdefg", "abd", "ez", "z", "ze", "abdd"];

        for entry in input.clone() {
            trie.insert(entry.chars());
        }

        for kv_pair in trie.clone().into_iter() {
            assert!(trie.contains(kv_pair.key.clone().chars()));
            let index = input
                .iter()
                .position(|&x| x == kv_pair.key.clone())
                .expect("should find it");
            input.remove(index);
        }
        assert!(input.is_empty())
    }

    #[test]
    fn it_iterates_over_populated_trie() {
        let mut trie = TrieString::<usize>::new();
        let mut input = vec!["abcdef", "abcdefg", "abd", "ez", "z", "ze", "abdd"];

        for entry in input.clone() {
            trie.insert(entry.chars());
        }

        for kv_pair in trie.iter() {
            assert!(trie.contains(kv_pair.key.clone().chars()));
            let index = input
                .iter()
                .position(|&x| x == kv_pair.key.clone())
                .expect("should find it");
            input.remove(index);
        }
        assert!(input.is_empty())
    }

    #[test]
    fn it_finds_in_owned_populated_trie() {
        static POPULATION_SIZE: usize = 1000;
        static SIZE: usize = 64;
        let mut trie = TrieString::<usize>::new();
        let mut input: HashSet<(String, Option<usize>)> = HashSet::new();
        for _i in 0..POPULATION_SIZE {
            let entry: Vec<char> = thread_rng()
                .sample_iter(&Alphanumeric)
                .take(thread_rng().gen_range(1..=SIZE))
                .map(char::from)
                .collect();
            let len = entry.len();
            input.insert((String::from_iter(entry.clone()), Some(len)));
            trie.insert_with_value(entry, Some(len));
        }
        let output: HashSet<(String, Option<usize>)> =
            trie.into_iter().map(|x| (x.key, x.value)).collect();
        assert_eq!(input, output);
    }

    #[test]
    fn it_finds_in_populated_trie() {
        static POPULATION_SIZE: usize = 1000;
        static SIZE: usize = 64;
        let mut trie = TrieString::<usize>::new();
        let mut input: HashSet<(String, Option<usize>)> = HashSet::new();
        for _i in 0..POPULATION_SIZE {
            let entry: Vec<char> = thread_rng()
                .sample_iter(&Alphanumeric)
                .take(thread_rng().gen_range(1..=SIZE))
                .map(char::from)
                .collect();
            let len = entry.len();
            input.insert((String::from_iter(entry.clone()), Some(len)));
            trie.insert_with_value(entry, Some(len));
        }
        let output: HashSet<(String, Option<usize>)> =
            trie.iter().map(|x| (x.key, x.value.cloned())).collect();
        assert_eq!(input, output);
    }
}
