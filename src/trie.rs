//! Provides a simple Trie implementation for storing keys composed of
//! sequences of atoms. A key may have an associated (optional) value.
//!
//! Atoms must support the TrieAtom trait. Atom values must support the
//! TrieValue trait.
//!
//! The interface relies on iterators to insert, remove, check for existence
//! of keys. Because the trie is based on the concept of atoms, then it
//! is up to the user to decide what kind of atoms to use to make most sense
//! of the keys we are storing.
//!
//! This flexibility can be really useful when string processing. Here are
//! three examples which show that we can work with keys of:
//!  - chars
//!  - grapheme clusters
//!  - &str ('words')
//!
//! depending on what type of atom granularity we wish to use when
//! interacting with our strings.
//!
//! Example 1
//! ```
//! use trying::trie::Trie;
//!
//! let mut trie = Trie::new();
//! let input = "abcdef".chars();
//! trie.insert_with_value(input.clone(), Some("abcdef".len()));
//!
//! // Anything which implements IntoIterator<Item=char> can now be used
//! // to interact with our Trie
//! assert!(trie.contains(input.clone())); // Clone the original iterator
//! assert!(trie.contains("abcdef".chars())); // Create a new iterator
//! assert!(trie.contains(['a', 'b', 'c', 'd', 'e', 'f'])); // Build an array, etc...
//! assert_eq!(trie.get(['a', 'b', 'c', 'd', 'e', 'f']), Some(&"abcdef".len())); // Get our value back
//! assert_eq!(trie.remove(input.clone()), Some("abcdef".len()));
//! assert!(!trie.contains(input));
//! ```
//!
//! Example 2
//! ```
//! use trying::trie::Trie;
//! use unicode_segmentation::UnicodeSegmentation;
//!
//! let mut trie: Trie<&str, usize> = Trie::new();
//! let s = "a̐éö̲\r\n";
//! let input = s.graphemes(true);
//! trie.insert(input.clone());
//! // Anything which implements IntoIterator<Item=&str> can now be used
//! // to interact with our Trie
//! assert!(trie.contains(input.clone()));
//! assert!(trie.remove(input.clone()).is_none());
//! assert!(!trie.contains(input));
//! ```
//!
//! Example 3
//! ```
//! use trying::trie::Trie;
//!
//! let mut trie = Trie::new();
//! let input = "the quick brown fox".split_whitespace();
//! trie.insert_with_value(input.clone(), Some(4));
//!
//! // Anything which implements IntoIterator<Item=&str> can now be used
//! // to interact with our Trie
//! assert!(trie.contains(input.clone()));
//! assert!(trie.contains_prefix("the quick brown".split_whitespace()));
//! assert_eq!(trie.remove(input.clone()), Some(4));
//! assert!(!trie.contains(input));
//! ```
//!
//! Here's an example of how we can iterate over our Trie. We use the
//! `FromIterator` trait to reconstruct our source key from the
//! vector of atoms which the iterator returns as the key.
//!
//! Example 4
//! ```
//! use std::iter::FromIterator;
//! use trying::trie::Trie;
//!
//! let mut trie = Trie::new();
//! let input = "the quick brown fox".split_whitespace();
//! trie.insert_with_value(input, Some(4));
//!
//! // Anything which implements IntoIterator<Item=&str> can now be used
//! // to interact with our Trie
//! for kv_pair in trie.into_iter() {
//!     println!("kv_pair: {:?}", kv_pair);
//!     assert_eq!("thequickbrownfox", String::from_iter(kv_pair.key));
//!     assert_eq!(kv_pair.value, Some(4));
//! }
//! ```
//! NB: Because we stripped all of the whitespace out when we built our
//! key, there is no whitespace in the re-assembled value. Until
//! `intersperse` is added to the std library, the simplest way to do
//! this right now is to use itertools. e.g.:
//! ```rustdoc
//! use itertools::Itertools;
//! assert_eq!("the quick brown fox", Itertools::intersperse(kv_pair.0.into_iter(), " ").collect::<String>());
//! ```
//!
//! Typical usages for this data structure:
//!  - Interning
//!  - Storing large numbers of keys with significant amounts of
//!    sub-key duplication
//!  - Prefix matching keys
//!  - ...

use crate::iterator::KeyValueRef;

#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};

/// Atoms which we wish to store in a Trie must implement
/// TrieAtom.
pub trait TrieAtom: Copy + Default + PartialEq + Ord {}

// Blanket implementation which satisfies the compiler
impl<A> TrieAtom for A
where
    A: Copy + Default + PartialEq + Ord,
{
    // Nothing to implement, since A already supports the other traits.
    // It has the functions it needs already
}

/// Values which we wish to store in a Trie must implement
/// TrieValue.
pub trait TrieValue: Default {}

// Blanket implementation which satisfies the compiler
impl<V> TrieValue for V
where
    V: Default,
{
    // Nothing to implement, since V already supports the other traits.
    // It has the functions it needs already
}

#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub(crate) struct AtomValue<A, V> {
    pub(crate) atom: A,
    pub(crate) value: Option<V>,
}

#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub(crate) struct Node<A, V> {
    pub(crate) children: Vec<Node<A, V>>,
    pub(crate) pair: AtomValue<A, V>,
    pub(crate) terminated: bool,
}

/// Stores a key of atoms as individual nodes.
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct Trie<A, V> {
    pub(crate) head: Node<A, V>,
    count: usize,
}

impl<A: TrieAtom, V: TrieValue> Node<A, V> {
    fn new(pair: AtomValue<A, V>) -> Self {
        Self {
            pair,
            ..Default::default()
        }
    }

    fn terminated(pair: AtomValue<A, V>) -> Self {
        Self {
            pair,
            terminated: true,
            ..Default::default()
        }
    }
}

impl<A: TrieAtom, V: TrieValue> Trie<A, V> {
    /// Create a new Trie.
    pub fn new() -> Self {
        Self {
            head: Node::default(),
            ..Default::default()
        }
    }

    /// Clear the Trie.
    pub fn clear(&mut self) {
        self.head = Node::default();
        self.count = 0;
    }

    /// Does the Trie contain the supplied key?
    pub fn contains<K: IntoIterator<Item = A>>(&self, key: K) -> bool {
        self.contains_internal(key, |n: &Node<A, V>| (n.terminated, None))
            .0
    }

    /// Does the Trie contain the supplied prefix?
    pub fn contains_prefix<P: IntoIterator<Item = A>>(&self, prefix: P) -> bool {
        self.contains_internal(prefix, |_| (true, None)).0
    }

    /// How many keys does the Trie contain?
    #[inline(always)]
    pub fn count(&self) -> usize {
        self.count
    }

    /// Get a reference to a key's associated value.
    pub fn get<K: IntoIterator<Item = A>>(&self, key: K) -> Option<&V> {
        self.contains_internal(key, |n: &Node<A, V>| (n.terminated, n.pair.value.as_ref()))
            .1
    }

    /// Get the longest common prefixes of the trie.
    ///
    /// This will be a Vec of prefixes. At least one, but possibly many more depending on
    /// the nature of the data contained within the trie.
    pub fn get_lcps<K: FromIterator<A>>(&self) -> Vec<K> {
        // The lcp will return a vec of longest prefixes
        let mut result = vec![];
        for node in self.head.children.iter() {
            let mut lcp: Vec<A> = vec![];
            let mut current_node = node;
            while current_node.children.len() == 1 && !current_node.terminated {
                lcp.push(current_node.pair.atom);
                current_node = current_node.children.get(0).unwrap();
            }
            lcp.push(current_node.pair.atom);
            result.push(lcp.into_iter().collect());
        }
        result
    }

    /// Insert the key (with a value of None) into the Trie. If the key is
    /// already present the value is updated to None. Returns the previously
    /// associated value.
    pub fn insert<K: IntoIterator<Item = A>>(&mut self, key: K) -> Option<V> {
        self.insert_with_value(key, None)
    }

    /// Insert the key and value into the Trie. If the key is already present
    /// the value is updated to the new value. Returns the previously
    /// associated value.
    pub fn insert_with_value<K: IntoIterator<Item = A>>(
        &mut self,
        key: K,
        value: Option<V>,
    ) -> Option<V> {
        let mut node = &mut self.head;
        let mut atoms = key.into_iter().peekable();
        let mut result = None;

        while let Some(atom) = atoms.next() {
            let last_idx = atoms.peek().is_none();

            let node_index = match node
                .children
                .iter_mut()
                .enumerate()
                .find(|(_i, x)| x.pair.atom == atom)
            {
                Some((i, mut n)) => {
                    if last_idx {
                        if !n.terminated {
                            self.count += 1;
                        }
                        result = n.pair.value.take();
                        n.pair.value = value;
                        n.terminated = true;
                        break;
                    }
                    i
                }
                None => {
                    if last_idx {
                        self.count += 1;
                        let new_node = Node::terminated(AtomValue { atom, value });
                        node.children.push(new_node);
                        break;
                    } else {
                        let new_node = Node::new(AtomValue { atom, value: None });
                        node.children.push(new_node);
                    };
                    node.children.len() - 1
                }
            };
            // Safe to unwrap here since we know we have these nodes in our children
            node = node.children.get_mut(node_index).unwrap();
        }
        result
    }

    /// Is the Trie empty?
    pub fn is_empty(&self) -> bool {
        self.head.children.is_empty()
    }

    /// Create an iterator over the Trie.
    pub fn iter(&self) -> impl Iterator<Item = KeyValueRef<'_, A, V>> {
        self.into_iter()
    }

    /// Create a sorted iterator over the Trie.
    pub fn iter_sorted(&self) -> impl Iterator<Item = KeyValueRef<'_, A, V>> {
        let mut v = self.into_iter().collect::<Vec<KeyValueRef<'_, A, V>>>();
        v.sort_by_cached_key(|x| x.key.clone());
        v.into_iter()
    }

    /// Remove the key from the Trie. If the key has an associated value, this
    /// is returned. If the key is not present or has an associated value of
    /// None, None is returned.
    pub fn remove<K: IntoIterator<Item = A>>(&mut self, key: K) -> Option<V> {
        let closure = |mut n: &mut Node<A, V>| {
            let present = n.terminated;
            n.terminated = false;
            (present, n.pair.value.take())
        };
        let result = self.contains_internal_mut(key, closure);
        if result.0 {
            self.count -= 1;
        }
        result.1
    }

    fn contains_internal<F: Fn(&Node<A, V>) -> (bool, Option<&V>), K: IntoIterator<Item = A>>(
        &self,
        key: K,
        f: F,
    ) -> (bool, Option<&V>) {
        let mut node = &self.head;
        let mut atoms = key.into_iter().peekable();
        while let Some(atom) = atoms.next() {
            let last_idx = atoms.peek().is_none();

            match node.children.iter().find(|x| x.pair.atom == atom) {
                Some(n) => {
                    if last_idx {
                        return f(n);
                    }
                    node = n;
                }
                None => {
                    break;
                }
            }
        }
        (false, None)
    }

    fn contains_internal_mut<
        F: Fn(&mut Node<A, V>) -> (bool, Option<V>),
        K: IntoIterator<Item = A>,
    >(
        &mut self,
        key: K,
        f: F,
    ) -> (bool, Option<V>) {
        let mut node = &mut self.head;
        let mut atoms = key.into_iter().peekable();
        while let Some(atom) = atoms.next() {
            let last_idx = atoms.peek().is_none();

            match node.children.iter_mut().find(|x| x.pair.atom == atom) {
                Some(n) => {
                    if last_idx {
                        return f(n);
                    }
                    node = n;
                }
                None => {
                    break;
                }
            }
        }
        (false, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use unicode_segmentation::UnicodeSegmentation;

    #[test]
    fn it_inserts_new_key() {
        let mut trie: Trie<char, usize> = Trie::new();
        trie.insert("abcdef".chars());
    }

    #[test]
    fn it_finds_exact_key() {
        let mut trie: Trie<char, usize> = Trie::new();
        let input = "abcdef".chars();
        trie.insert(input.clone());
        assert!(trie.contains(input));
    }

    #[test]
    fn it_cannot_find_longer_key() {
        let mut trie: Trie<char, usize> = Trie::new();
        let input = "abcdef".chars();
        let long_input = "abcdefg".chars();
        trie.insert(input);
        assert!(!trie.contains(long_input));
    }

    #[test]
    fn it_cannot_find_shorter_key() {
        let mut trie: Trie<char, usize> = Trie::new();
        let input = "abcdef".chars();
        let short_input = "abcde".chars();
        trie.insert(input);
        assert!(!trie.contains(short_input));
    }

    #[test]
    fn it_can_find_multiple_overlapping_keys() {
        let mut trie: Trie<char, usize> = Trie::new();
        let input = "abcdef".chars();
        trie.insert(input.clone());
        let short_input = "abc".chars();
        trie.insert(short_input.clone());
        assert!(trie.contains(short_input));
        assert!(trie.contains(input));
    }

    #[test]
    fn it_can_find_prefix_keys() {
        let mut trie: Trie<char, usize> = Trie::new();
        let input = "abcdef".chars();
        let short_input = "abc".chars();
        trie.insert(input);
        assert!(trie.contains_prefix(short_input));
    }

    #[test]
    fn it_can_remove_a_present_key() {
        let mut trie: Trie<char, usize> = Trie::new();
        let input = "abcdef".chars();
        trie.insert(input.clone());
        assert!(trie.contains(input.clone()));
        assert!(trie.remove(input.clone()).is_none());
        assert!(!trie.contains(input));
    }

    #[test]
    fn it_can_remove_a_missing_key() {
        let mut trie: Trie<char, usize> = Trie::new();
        let input = "abcdef".chars();
        assert!(trie.remove(input.clone()).is_none());
        assert!(!trie.contains(input));
    }

    #[test]
    fn it_can_return_previously_inserted_value() {
        let mut trie: Trie<char, usize> = Trie::new();
        let input = "abcdef".chars();
        trie.insert_with_value(input.clone(), Some(666));
        assert_eq!(trie.insert_with_value(input.clone(), Some(667)), Some(666));
        assert_eq!(trie.remove(input.clone()), Some(667));
        assert_eq!(trie.remove(input.clone()), None);
        assert!(!trie.contains(input));
    }

    #[test]
    fn it_can_create_an_empty_trie() {
        let trie: Trie<char, usize> = Trie::new();
        assert!(trie.is_empty());
    }

    #[test]
    fn it_can_clear_a_trie() {
        let mut trie: Trie<char, usize> = Trie::new();
        let input = "abcdef".chars();
        trie.insert(input.clone());
        trie.clear();
        assert!(trie.is_empty());
        assert!(!trie.contains(input));
    }

    #[test]
    fn it_can_count_entries() {
        let mut trie: Trie<char, usize> = Trie::new();
        let input = "abcdef".chars();
        trie.insert(input.clone());
        assert_eq!(1, trie.count());
        trie.insert(input.clone());
        trie.insert(input.clone());
        assert_eq!(1, trie.count());
        trie.remove(input.clone());
        assert_eq!(0, trie.count());
        trie.clear();
        assert_eq!(0, trie.count());
        assert!(trie.is_empty());
        assert!(!trie.contains(input));
    }

    // usize unit tests
    #[test]
    fn it_inserts_new_usize_key() {
        let mut trie: Trie<usize, usize> = Trie::new();
        let input: Vec<usize> = vec![0, 1, 2, 3, 4, 5, 6];
        trie.insert(input);
    }

    #[test]
    fn it_finds_exact_usize_key() {
        let mut trie: Trie<usize, usize> = Trie::new();
        let input = [0, 1, 2, 3, 4, 5, 6];
        trie.insert(input);
        assert!(trie.contains(input));
    }

    #[test]
    fn it_cannot_find_short_usize_key() {
        let mut trie: Trie<usize, usize> = Trie::new();
        let input = [0, 1, 2, 3, 4, 5, 6];
        let input_short = [0, 1, 2, 3, 4, 5];
        trie.insert(input);
        assert!(!trie.contains(input_short));
    }

    // grapheme cluster unit test
    #[test]
    fn it_can_process_grapheme_clusters() {
        let mut trie: Trie<&str, bool> = Trie::new();
        let s = "a̐éö̲\r\n";
        let input = s.graphemes(true);
        trie.insert(input.clone());
        assert!(trie.contains(input.clone()));
        assert!(trie.remove(input.clone()).is_none());
        assert!(!trie.contains(input));
    }

    // &str unit test
    #[test]
    fn it_can_process_str_clusters() {
        let mut trie = Trie::new();
        let input = "the quick brown fox".split_whitespace();
        trie.insert_with_value(input.clone(), Some(5));
        assert_eq!(trie.get(input.clone()), Some(&5));
        assert!(trie.contains(input.clone()));
        assert!(trie.remove(input.clone()).is_some());
        assert!(!trie.contains(input));
    }

    // serialization test
    #[test]
    fn it_serializes_trie_to_json() {
        let mut t1: Trie<usize, usize> = Trie::new();
        let input = [0, 1, 2, 3, 4, 5, 6];
        t1.insert(input);
        // Round trip via serde to create a new trie and then
        // check for equality
        let t_str = serde_json::to_string(&t1).expect("serializing");
        let t2: Trie<usize, usize> = serde_json::from_str(&t_str).expect("deserializing");
        assert_eq!(t1, t2);
    }
    #[test]
    fn it_can_find_lcp() {
        let input = vec![
            "code",
            "coder",
            "coding",
            "codable",
            "codec",
            "codecs",
            "coded",
            "codeless",
            "codependence",
            "codependency",
            "codependent",
            "codependents",
            "codes",
            "a",
            "codesign",
            "codesigned",
            "codeveloped",
            "codeveloper",
            "abc",
            "codex",
            "codify",
            "codiscovered",
            "codrive",
            "abz",
        ];
        let mut trie: Trie<char, ()> = Trie::new();
        for entry in input {
            trie.insert(entry.chars());
        }
        assert_eq!(vec!["cod", "a"], trie.get_lcps::<String>());
    }

    #[test]
    fn it_can_find_lcp_usize() {
        let input = vec![
            vec![1, 11, 111, 1111],
            vec![1, 11, 111, 1111, 11112],
            vec![1, 11, 111, 1111, 11113],
        ];
        let mut trie: Trie<usize, ()> = Trie::new();
        for entry in input {
            trie.insert(entry);
        }
        assert_eq!(vec![vec![1, 11, 111, 1111]], trie.get_lcps::<Vec<usize>>());
    }

    #[test]
    fn it_can_iter_sorted() {
        let input = vec![
            "lexicographic",
            "sorting",
            "of",
            "a",
            "set",
            "of",
            "keys",
            "can",
            "be",
            "accomplished",
            "with",
            "a",
            "simple",
            "trie",
            "based",
            "algorithm",
            "we",
            "insert",
            "all",
            "keys",
            "in",
            "a",
            "trie",
            "output",
            "all",
            "keys",
            "in",
            "the",
            "trie",
            "by",
            "means",
            "of",
            "preorder",
            "traversal",
            "which",
            "results",
            "in",
            "output",
            "that",
            "is",
            "in",
            "lexicographically",
            "increasing",
            "order",
            "preorder",
            "traversal",
            "is",
            "a",
            "kind",
            "of",
            "depth",
            "first",
            "traversal",
        ];
        let mut trie: Trie<char, ()> = Trie::new();
        for entry in input {
            trie.insert(entry.chars());
        }
        let sorted_words: Vec<String> = trie
            .iter_sorted()
            .into_iter()
            .map(|x| x.key.iter().collect())
            .collect();
        println!("sorted_words: {:?}", sorted_words);
    }

    #[test]
    fn it_can_find_maximum_occurring_entry() {
        let input = vec![
            "code",
            "coder",
            "coding",
            "codable",
            "codec",
            "codecs",
            "coded",
            "codeless",
            "codec",
            "codecs",
            "codependence",
            "codex",
            "codify",
            "codependents",
            "codes",
            "code",
            "coder",
            "codesign",
            "codec",
            "codeveloper",
            "codrive",
            "codec",
            "codecs",
            "codiscovered",
        ];
        let mut trie: Trie<char, usize> = Trie::new();
        for entry in input {
            let ch = entry.chars();
            match trie.get(ch.clone()) {
                Some(v) => trie.insert_with_value(ch, Some(v + 1)),
                None => trie.insert_with_value(ch, Some(1)),
            };
        }
        let mut answer = None;
        let mut highest = 0;
        for entry in trie.iter() {
            if let Some(&v) = entry.value {
                if v > highest {
                    highest = v;
                    answer = Some(entry.key.clone());
                }
            }
        }
        // There should be 4 "codec"
        assert_eq!(highest, 4);
        assert_eq!(answer, Some(vec!['c', 'o', 'd', 'e', 'c']));
    }
}
