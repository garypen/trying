//! Provides Trie iterators.
//!
use crate::trie::{Node, Trie, TrieAtom, TrieValue};

/// Iterator Item
#[derive(Debug)]
pub struct KeyValue<A, V> {
    pub key: Vec<A>,
    pub value: Option<V>,
}

/// Iterator Item
#[derive(Debug)]
pub struct KeyValueRef<'a, A, V> {
    pub key: Vec<A>,
    pub value: Option<&'a V>,
}

/// Iterator over a Trie.
#[derive(Debug)]
pub struct TrieIntoIterator<A, V> {
    results: Vec<KeyValue<A, V>>,
    backtrack: usize,
    nodes: Vec<Node<A, V>>,
}

impl<A: TrieAtom, V: TrieValue> IntoIterator for Trie<A, V> {
    type Item = KeyValue<A, V>;
    type IntoIter = TrieIntoIterator<A, V>;

    fn into_iter(self) -> Self::IntoIter {
        let mut results: Vec<Self::Item> = vec![];

        let mut nodes = vec![self.head];

        // Create our seed column and results
        Trie::make_column(&mut nodes);
        Trie::create_results(&mut results, &mut nodes[1..]);

        results.reverse();
        TrieIntoIterator {
            results,
            backtrack: 0,
            nodes,
        }
    }
}

impl<A: TrieAtom, V: TrieValue> Iterator for TrieIntoIterator<A, V> {
    type Item = KeyValue<A, V>;

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
                    Trie::make_column(&mut self.nodes);
                    Trie::create_results(&mut self.results, &mut self.nodes[1..]);
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
pub struct TrieRefIntoIterator<'a, A: TrieAtom, V: TrieValue> {
    results: Vec<KeyValueRef<'a, A, V>>,
    backtrack: usize,
    nodes: Vec<NodeRef<'a, A, V>>,
}

impl<'a, A: TrieAtom, V: TrieValue> Iterator for TrieRefIntoIterator<'a, A, V> {
    type Item = KeyValueRef<'a, A, V>;

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
                    Trie::make_tracked_column(&mut self.nodes);
                    Trie::create_tracked_results(&mut self.results, &self.nodes[1..]);
                }
                self.results.reverse();
                self.results.pop()
            }
        }
    }
}

// Iterator
impl<'a, A: TrieAtom, V: TrieValue> IntoIterator for &'a Trie<A, V> {
    type Item = KeyValueRef<'a, A, V>;
    type IntoIter = TrieRefIntoIterator<'a, A, V>;

    fn into_iter(self) -> Self::IntoIter {
        let mut results: Vec<Self::Item> = vec![];

        let mut nodes = vec![NodeRef(&self.head, 0)];

        // Create our seed column and results
        Trie::make_tracked_column(&mut nodes);
        Trie::create_tracked_results(&mut results, &nodes[1..]);

        results.reverse();
        TrieRefIntoIterator {
            results,
            backtrack: 0,
            nodes,
        }
    }
}

// Useful utility functions for building iterator output
impl<'a, A: TrieAtom, V: TrieValue> Trie<A, V> {
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
                nodes.push(NodeRef(child, 0));
            } else {
                break;
            }
        }
    }

    fn create_results(results: &mut Vec<KeyValue<A, V>>, nodes: &mut [Node<A, V>]) {
        let mut current = vec![];
        for node in nodes {
            current.push(node.pair.atom);
            if node.terminated {
                results.push(KeyValue {
                    key: current.clone(),
                    value: node.pair.value.take(),
                });
            }
        }
    }

    fn create_tracked_results<'b: 'a>(
        results: &mut Vec<KeyValueRef<'b, A, V>>,
        nodes: &'a [NodeRef<'b, A, V>],
    ) {
        let mut current = vec![];
        for node in nodes {
            current.push(node.0.pair.atom);
            if node.0.terminated {
                results.push(KeyValueRef {
                    key: current.clone(),
                    value: node.0.pair.value.as_ref(),
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use itertools::Itertools;
    use rand::{distributions::Alphanumeric, thread_rng, Rng};
    use std::iter::FromIterator;

    #[test]
    fn it_iterates_over_empty_trie() {
        let trie: Trie<char, usize> = Trie::new();
        for kv_pair in trie.into_iter() {
            println!("kv_pair: {:?}", kv_pair);
        }
    }

    #[test]
    fn it_iterates_and_re_assembles_trie() {
        let mut trie = Trie::new();
        let input = "the quick brown fox".split_whitespace();
        trie.insert_with_value(input.clone(), Some(4));

        for kv_pair in trie.into_iter() {
            println!("kv_pair: {:?}", kv_pair);
            assert_eq!(
                "the quick brown fox",
                Itertools::intersperse(kv_pair.key.into_iter(), " ").collect::<String>()
            );
        }
    }

    #[test]
    fn it_iterates_over_owned_populated_trie() {
        let mut trie: Trie<char, usize> = Trie::new();
        let input = "abcdef".chars();
        trie.insert(input);
        let input = "abcdefg".chars();
        trie.insert(input);
        let input = "abd".chars();
        trie.insert(input);
        let input = "ez".chars();
        trie.insert(input);
        let input = "z".chars();
        trie.insert(input);
        let input = "ze".chars();
        trie.insert(input);
        let input = "abdd".chars();
        trie.insert(input);
        for kv_pair in trie.into_iter() {
            println!("kv_pair: {:?}", kv_pair);
            println!("string: {:?}", String::from_iter(kv_pair.key));
        }
    }

    #[test]
    fn it_iterates_over_populated_trie() {
        let mut trie: Trie<char, usize> = Trie::new();
        let input = "abcdef".chars();
        trie.insert(input);
        let input = "abcdefg".chars();
        trie.insert(input);
        let input = "abd".chars();
        trie.insert(input);
        let input = "ez".chars();
        trie.insert(input);
        let input = "z".chars();
        trie.insert(input);
        let input = "ze".chars();
        trie.insert(input);
        let input = "abdd".chars();
        trie.insert(input);
        for kv_pair in (&trie).iter() {
            println!("kv_pair: {:?}", kv_pair);
            println!("string: {:?}", String::from_iter(kv_pair.key));
        }
    }

    #[test]
    fn it_finds_in_owned_populated_trie() {
        static POPULATION_SIZE: usize = 1000;
        static SIZE: usize = 64;
        let mut trie: Trie<char, usize> = Trie::new();
        let mut searches: Vec<Vec<char>> = vec![];
        for _i in 0..POPULATION_SIZE {
            let entry: Vec<char> = thread_rng()
                .sample_iter(&Alphanumeric)
                .take(thread_rng().gen_range(1..=SIZE))
                .map(char::from)
                .collect();
            searches.push(entry.clone());
            let len = entry.len();
            trie.insert_with_value(entry, Some(len));
        }
        for entry in &searches {
            let mut iterator = trie.clone().into_iter();
            assert_eq!(
                Some(entry.clone()),
                iterator.find(|x| x.key == *entry).map(|x| x.key)
            );
        }
    }

    #[test]
    fn it_finds_in_populated_trie() {
        static POPULATION_SIZE: usize = 1000;
        static SIZE: usize = 64;
        let mut trie: Trie<char, usize> = Trie::new();
        let mut searches: Vec<Vec<char>> = vec![];
        for _i in 0..POPULATION_SIZE {
            let entry: Vec<char> = thread_rng()
                .sample_iter(&Alphanumeric)
                .take(thread_rng().gen_range(1..=SIZE))
                .map(char::from)
                .collect();
            searches.push(entry.clone());
            let len = entry.len();
            trie.insert_with_value(entry, Some(len));
        }
        for entry in &searches {
            let mut iterator = trie.iter();
            assert_eq!(
                Some(entry.clone()),
                iterator.find(|x| x.key == *entry).map(|x| x.key)
            );
        }
    }
}
