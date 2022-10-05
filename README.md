# trying
Provides a simple Trie implementation for storing "keys" composed of "atoms".

The trie imposes restrictions on the key, atom and value types:
 - keys must be: Clone + Default + Ord + FromIterator<A> (aggregate trait: TrieKey)
 - atoms must be: Copy + Default + PartialEq + Ord (aggregate trait: TrieAtom)
 - values must be: Default (aggregate trait: TrieValue)

(where A represents the Atom type that the key will be represented as)

With these restrictions in place, the trie implements a reasonably efficient
mechanism for:
 - prefix matching
 - representing large quantities of data with common prefixes
 - finding shortest unique prefix
 - finding alternative values
 - finding longest common prefixes

For Example:

```
use trying::trie::TrieVec;
use unicode_segmentation::UnicodeSegmentation;

// Declare a trie which will store &str keys
// with usize values.
let mut trie = TrieVec::<&str, usize>::new();
let s = "a̐éö̲\r\n";
let input = s.graphemes(true);
// Insert our graphemes into the trie
trie.insert(input.clone());
// Anything which implements IntoIterator<Item=&str> can now be used
// to interact with our Trie
assert!(trie.contains(input.clone()));
assert!(trie.remove(input.clone()).is_none());
assert!(!trie.contains(input));
```

If you don't need prefix matching, then a HashMap is almost always a better
choice than a trie...

[![Crates.io](https://img.shields.io/crates/v/trying.svg)](https://crates.io/crates/trying)

[API Docs](https://docs.rs/trying/latest/trying)

## Installation

```toml
[dependencies]
trying = "0.4"
```

[Features are available](https://github.com/garypen/trying/blob/main/Cargo.toml#L19).

## License

Apache 2.0 licensed. See LICENSE for details.
