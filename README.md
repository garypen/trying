# trying
Provides a simple Trie implementation for storing "keys" composed of "atoms".

The trie imposes restrictions on the key and value types:
 - keys must be: Clone + Default + Ord + FromIterator<A> (aggregate trait: TrieKey)
 - atoms must be: Copy + Default + PartialEq + Ord (aggregate trait: TrieAtom)
 - values must be: Default (aggregate trait: TrieValue)

(where A represents the Atom type that the key will be represented as)

With these restrictions in place, the trie implements a reasonably efficient
mechanism for:
 - prefix matching
 - representing large quantities of data with common prefixes

If you don't need prefix matching, then a HashMap is almost always a better
choice than a trie...

[![Crates.io](https://img.shields.io/crates/v/trying.svg)](https://crates.io/crates/trying)

[API Docs](https://docs.rs/trying/latest/trying)

## Installation

```toml
[dependencies]
trying = "0.3"
```

[Features are available](https://github.com/garypen/trying/blob/main/Cargo.toml#L19).

## License

Apache 2.0 licensed. See LICENSE for details.
