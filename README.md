# trying
Provides a simple Trie implementation for storing "keys" composed of "atoms".

The trie imposes restrictions on the key and value types:
 - keys must be: Copy + Default + PartialEq (aggregate trait: TrieAtom)
 - values must be: Default (aggregate trait: TrieValue)

With these restrictions in place, the trie implements a reasonably efficient
mechanism for:
 - prefix matching
 - representing large quantities of data with common prefixes

If you don't need prefix matching, then a HashMap is almost always a better
choice than a trie...

## Installation

```toml
[dependencies]
trying = "0.1"
```

[Features are available](https://github.com/garypen/trying/blob/main/Cargo.toml#L19).

## License

Apache 2.0 licensed. See LICENSE for details.
