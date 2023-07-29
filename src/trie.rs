//! Trie implementation
//!
//! Here are three examples which show that we can work with keys composed
//! from different types of atoms:
//!  - chars
//!  - grapheme clusters
//!  - &str ('words')
//!
//! depending on what type of atom granularity we wish to use when
//! interacting with our strings.
//!
//! Example 1
//! ```
//! use trying::trie::TrieString;
//!
//! let mut trie = TrieString::<usize>::new();
//! let input = "abcdef".chars();
//! trie.insert_with_value(input.clone(), Some("abcdef".len()));
//!
//! // Anything which implements IntoIterator<Item=char> can now be used
//! // to interact with our Trie
//! assert!(trie.contains(input.clone())); // The original iterator
//! assert!(trie.contains("abcdef".chars())); // A new iterator
//! assert!(trie.contains(['a', 'b', 'c', 'd', 'e', 'f'])); // Build an array, etc...
//! assert_eq!(trie.get(['a', 'b', 'c', 'd', 'e', 'f']), Some(&"abcdef".len())); // Get our value back
//! assert_eq!(trie.remove(input.clone()), Some("abcdef".len()));
//! assert!(!trie.contains(input));
//! ```
//!
//! Example 2
//! ```
//! use trying::trie::TrieVec;
//! use unicode_segmentation::UnicodeSegmentation;
//!
//! let mut trie = TrieVec::<&str, usize>::new();
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
//! use trying::trie::TrieVec;
//!
//! let mut trie = TrieVec::<&str, usize>::new();
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

use crate::iterator::KeyValueRef;

use std::iter::FromIterator;

#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};

// Some types of trie are likely to be very common, so let's define some types for them

/// Convenience for typical String processing. Equivalent to `Trie<String, char, A>`
pub type TrieString<V> = Trie<String, char, V>;

/// Convenience for typical processing. Equivalent to `Trie<Vec<A>, A, V>`
pub type TrieVec<A, V> = Trie<Vec<A>, A, V>;

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

/// Keys which we wish to store in a Trie must implement
/// TrieKey.
pub trait TrieKey<A>: Clone + Default + Ord + FromIterator<A> {}

// Blanket implementation which satisfies the compiler
impl<A, K> TrieKey<A> for K
where
    K: Clone + Default + Ord + FromIterator<A>,
{
    // Nothing to implement, since K already supports the other traits.
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
pub struct Trie<K, A, V> {
    pub(crate) head: Node<A, V>,
    count: usize,
    phantom: std::marker::PhantomData<K>,
    atoms: usize,
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

impl<K: TrieKey<A>, A: TrieAtom, V: TrieValue> Trie<K, A, V> {
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
        self.atoms = 0;
    }

    /// Does the Trie contain the supplied key?
    pub fn contains<I: IntoIterator<Item = A>>(&self, key: I) -> bool {
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

    /// How many atoms does the Trie contain?
    #[inline(always)]
    pub fn atoms(&self) -> usize {
        self.atoms
    }

    /// Get a reference to a key's associated value.
    pub fn get<I: IntoIterator<Item = A>>(&self, key: I) -> Option<&V> {
        self.contains_internal(key, |n: &Node<A, V>| (n.terminated, n.pair.value.as_ref()))
            .1
    }

    /// Get alternative to a supplied key
    pub fn get_alternatives<I: Clone + IntoIterator<Item = A>>(
        &self,
        key: I,
        limit: usize,
    ) -> Vec<K> {
        // Search for our key, if we find it, then just return it
        if self
            .contains_internal(key.clone(), |n: &Node<A, V>| (n.terminated, None))
            .0
        {
            vec![K::from_iter(key)]
        } else {
            let mut new_key: Vec<A> = vec![];

            let mut atoms = key.into_iter().peekable();
            while let Some(atom) = atoms.next() {
                let last_idx = atoms.peek().is_none();
                if last_idx {
                    break;
                } else {
                    new_key.push(atom);
                }
            }
            let mut base = vec![];

            let mut node = &self.head;

            for atom in new_key.into_iter() {
                match node.children.iter().find(|x| x.pair.atom == atom) {
                    Some(n) => {
                        base.push(n.pair.atom);
                        node = n;
                    }
                    None => {
                        break;
                    }
                }
            }

            // Now start to build our list of alternatives
            let mut alternatives = vec![];

            // Logic is convoluted. May improve in future...
            'outer: loop {
                for mut child in node.children.iter().take(limit) {
                    // Build an alternative
                    let mut alternative = base.clone();
                    while !child.terminated && !child.children.is_empty() {
                        alternative.push(child.pair.atom);
                        child = &child.children[0];
                    }
                    alternative.push(child.pair.atom);

                    // Evaluate the alternative
                    let candidate = K::from_iter(alternative);
                    if !alternatives.contains(&candidate) {
                        alternatives.push(candidate);
                        // Have we reached our specified limit?
                        if alternatives.len() == limit {
                            break 'outer;
                        }
                    }
                }
                // Have we run out of alternatives to consider without reaching
                // our specified limit
                if node.children.is_empty() {
                    break;
                } else {
                    node = &node.children[0];
                    base.push(node.pair.atom);
                }
            }
            alternatives
        }
    }

    /// Get the longest common prefixes of the trie.
    ///
    /// This will be a Vec of prefixes. At least one, but possibly many more depending on
    /// the nature of the data contained within the trie.
    pub fn get_lcps(&self) -> Vec<K> {
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

    /// Get the shortest unique prefix for a key
    ///
    /// This will be an option, since:
    ///  - the key may not exist in the trie
    ///  - there may not be a unique prefix.
    pub fn get_sup<I: IntoIterator<Item = A>>(&self, key: I) -> Option<K> {
        let mut node = &self.head;

        let mut nodes = vec![];
        let mut master = vec![];

        for atom in key {
            match node.children.iter().find(|x| x.pair.atom == atom) {
                Some(n) => {
                    nodes.push((n, node.children.len()));
                    master.push(n.pair.atom);
                    node = n;
                }
                None => {
                    // Key isn't in the trie
                    return None;
                }
            }
        }

        // Now, remove the correct number of nodes from our key to find the sup.
        // The logic is to search backwards through our set of nodes until
        // we find one with !1 child.
        let mut remove = 0;
        nodes.reverse();

        for node in nodes {
            if node.1 == 1 {
                remove += 1;
            } else {
                master.truncate(master.len() - remove);
                return Some(master.into_iter().collect());
            }
        }
        None
    }

    /// Insert the key (with a value of None) into the Trie. If the key is
    /// already present the value is updated to None. Returns the previously
    /// associated value.
    pub fn insert<I: IntoIterator<Item = A>>(&mut self, key: I) -> Option<V> {
        self.insert_with_value(key, None)
    }

    /// Insert the key and value into the Trie. If the key is already present
    /// the value is updated to the new value. Returns the previously
    /// associated value.
    pub fn insert_with_value<I: IntoIterator<Item = A>>(
        &mut self,
        key: I,
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
                        self.atoms += 1;
                        node.children.push(new_node);
                        break;
                    } else {
                        let new_node = Node::new(AtomValue { atom, value: None });
                        self.atoms += 1;
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
    pub fn iter(&self) -> impl Iterator<Item = KeyValueRef<'_, K, A, V>> {
        self.into_iter()
    }

    /// Create a sorted iterator over the Trie.
    pub fn iter_sorted(&self) -> impl Iterator<Item = KeyValueRef<'_, K, A, V>> {
        let mut v = self.into_iter().collect::<Vec<KeyValueRef<'_, K, A, V>>>();
        v.sort_by_key(|x| x.key.clone());
        v.into_iter()
    }

    /// Remove the key from the Trie. If the key has an associated value, this
    /// is returned. If the key is not present or has an associated value of
    /// None, None is returned.
    pub fn remove<I: IntoIterator<Item = A>>(&mut self, key: I) -> Option<V> {
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

    fn contains_internal<F: Fn(&Node<A, V>) -> (bool, Option<&V>), I: IntoIterator<Item = A>>(
        &self,
        key: I,
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
        I: IntoIterator<Item = A>,
    >(
        &mut self,
        key: I,
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
        let mut trie = TrieString::<usize>::new();
        trie.insert("abcdef".chars());
    }

    #[test]
    fn it_finds_exact_key() {
        let mut trie = TrieString::<usize>::new();
        let input = "abcdef".chars();
        trie.insert(input.clone());
        assert!(trie.contains(input));
    }

    #[test]
    fn it_cannot_find_longer_key() {
        let mut trie = TrieString::<usize>::new();
        let input = "abcdef".chars();
        let long_input = "abcdefg".chars();
        trie.insert(input);
        assert!(!trie.contains(long_input));
    }

    #[test]
    fn it_cannot_find_shorter_key() {
        let mut trie = TrieString::<usize>::new();
        let input = "abcdef".chars();
        let short_input = "abcde".chars();
        trie.insert(input);
        assert!(!trie.contains(short_input));
    }

    #[test]
    fn it_can_find_multiple_overlapping_keys() {
        let mut trie = TrieString::<usize>::new();
        let input = "abcdef".chars();
        trie.insert(input.clone());
        let short_input = "abc".chars();
        trie.insert(short_input.clone());
        assert!(trie.contains(short_input));
        assert!(trie.contains(input));
    }

    #[test]
    fn it_can_find_prefix_keys() {
        let mut trie = TrieString::<usize>::new();
        let input = "abcdef".chars();
        let short_input = "abc".chars();
        trie.insert(input);
        assert!(trie.contains_prefix(short_input));
    }

    #[test]
    fn it_can_remove_a_present_key() {
        let mut trie = TrieString::<usize>::new();
        let input = "abcdef".chars();
        trie.insert(input.clone());
        assert!(trie.contains(input.clone()));
        assert!(trie.remove(input.clone()).is_none());
        assert!(!trie.contains(input));
    }

    #[test]
    fn it_can_remove_a_missing_key() {
        let mut trie = TrieString::<usize>::new();
        let input = "abcdef".chars();
        assert!(trie.remove(input.clone()).is_none());
        assert!(!trie.contains(input));
    }

    #[test]
    fn it_can_return_previously_inserted_value() {
        let mut trie = TrieString::<usize>::new();
        let input = "abcdef".chars();
        trie.insert_with_value(input.clone(), Some(666));
        assert_eq!(trie.insert_with_value(input.clone(), Some(667)), Some(666));
        assert_eq!(trie.remove(input.clone()), Some(667));
        assert_eq!(trie.remove(input.clone()), None);
        assert!(!trie.contains(input));
    }

    #[test]
    fn it_can_create_an_empty_trie() {
        let trie = TrieString::<usize>::new();
        assert!(trie.is_empty());
    }

    #[test]
    fn it_can_clear_a_trie() {
        let mut trie = TrieString::<usize>::new();
        let input = "abcdef".chars();
        trie.insert(input.clone());
        trie.clear();
        assert!(trie.is_empty());
        assert!(!trie.contains(input));
    }

    #[test]
    fn it_can_count_entries() {
        let mut trie = TrieString::<usize>::new();
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
        let mut trie = TrieVec::<usize, usize>::new();
        let input: Vec<usize> = vec![0, 1, 2, 3, 4, 5, 6];
        trie.insert(input);
    }

    #[test]
    fn it_finds_exact_usize_key() {
        let mut trie = TrieVec::<usize, usize>::new();
        let input = [0, 1, 2, 3, 4, 5, 6];
        trie.insert(input);
        assert!(trie.contains(input));
    }

    #[test]
    fn it_cannot_find_short_usize_key() {
        let mut trie = TrieVec::<usize, usize>::new();
        let input = [0, 1, 2, 3, 4, 5, 6];
        let input_short = [0, 1, 2, 3, 4, 5];
        trie.insert(input);
        assert!(!trie.contains(input_short));
    }

    // grapheme cluster unit test
    #[test]
    fn it_can_process_grapheme_clusters() {
        let mut trie = TrieVec::<&str, bool>::new();
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
        let mut trie = TrieVec::<&str, usize>::new();
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
        let mut t1 = TrieVec::<usize, usize>::new();
        let input = [0, 1, 2, 3, 4, 5, 6];
        t1.insert(input);
        // Round trip via serde to create a new trie and then
        // check for equality
        let t_str = serde_json::to_string(&t1).expect("serializing");
        let t2: TrieVec<usize, usize> = serde_json::from_str(&t_str).expect("deserializing");
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
        let mut trie = TrieString::<()>::new();
        for entry in input {
            trie.insert(entry.chars());
        }
        assert_eq!(vec!["cod", "a"], trie.get_lcps());
    }

    #[test]
    fn it_can_find_lcp_usize() {
        let input = vec![
            vec![1, 11, 111, 1111],
            vec![1, 11, 111, 1111, 11112],
            vec![1, 11, 111, 1111, 11113],
        ];
        let mut trie = TrieVec::<usize, ()>::new();
        for entry in input {
            trie.insert(entry);
        }
        assert_eq!(vec![vec![1, 11, 111, 1111]], trie.get_lcps());
    }

    #[test]
    fn it_can_find_sups_that_exist() {
        let input = vec!["AND", "BONFIRE", "BOOL", "CASE", "CATCH", "CHAR"];
        let output = vec!["A", "BON", "BOO", "CAS", "CAT", "CH"];
        let mut trie = TrieString::<()>::new();

        for entry in input.clone() {
            trie.insert(entry.chars());
        }

        for (inn, out) in input.into_iter().zip(output.into_iter()) {
            assert_eq!(trie.get_sup(inn.to_string().chars()), Some(out.to_string()));
        }
    }

    #[test]
    fn it_cannot_find_sups_that_have_prefixes() {
        let base = vec!["AND", "BONFIRE", "BOOL", "CASE", "CATCH", "CHAR"];
        let input = vec!["ANDY", "BONFIREY", "BOOLY", "CASEY", "CATCHY", "CHARY"];
        let output = vec![None; 6];
        let mut trie = TrieString::<()>::new();

        for entry in base {
            trie.insert(entry.chars());
        }

        for (inn, out) in input.into_iter().zip(output.into_iter()) {
            assert_eq!(trie.get_sup(inn.to_string().chars()), out);
        }
    }

    #[test]
    fn it_cannot_find_sups_that_are_just_wrong() {
        let base = vec!["AND", "BONFIRE", "BOOL", "CASE", "CATCH", "CHAR"];
        let input = vec!["WHAT", "IS", "THIS", "TEST", "ALL", "ABOUT"];
        let output = vec![None; 6];
        let mut trie = TrieString::<()>::new();

        for entry in base {
            trie.insert(entry.chars());
        }

        for (inn, out) in input.into_iter().zip(output.into_iter()) {
            assert_eq!(trie.get_sup(inn.to_string().chars()), out);
        }
    }

    #[test]
    fn it_can_iter_sorted() {
        let mut input = vec![
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
        let mut trie = TrieString::<()>::new();
        for entry in &input {
            trie.insert(entry.chars());
        }
        let sorted_words: Vec<String> = trie.iter_sorted().map(|x| x.key).collect();
        // Sort our input and deduplicate it
        input.sort();
        input.dedup();
        assert_eq!(input, sorted_words);
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
        let mut trie = TrieString::<usize>::new();
        for entry in input {
            let ch = entry.chars();
            let value = match trie.get(ch.clone()) {
                Some(v) => v + 1,
                None => 1,
            };
            trie.insert_with_value(ch, Some(value));
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
        assert_eq!(answer, Some("codec".to_string()));
    }

    #[test]
    fn it_can_find_alternatives() {
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
        let mut trie = TrieString::<()>::new();
        for entry in input {
            let ch = entry.chars();
            trie.insert(ch);
        }
        assert_eq!(
            trie.get_alternatives("codg".chars(), 5),
            ["code", "coding", "codable", "codrive", "coder"]
        )
    }
}
