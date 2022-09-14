//! Provides a simple Trie implementation for storing keys composed of
//! a [`std::vec::Vec`] of atoms. A key may have an associated
//! value.
//!
//! Keys must support the [`crate::trie::TrieKey`] trait. Atoms must support
//! the [`crate::trie::TrieAtom`] trait. Atom values must support the
//! [`crate::trie::TrieValue`] trait.
//!
//! The interface relies on iterators to insert, remove, check for existence
//! of keys. Because the trie is based on the concept of atoms, then it
//! is up to the user to decide what kind of atoms to use to make most sense
//! of the keys we are storing. This flexibility can be really useful when
//! string processing: (atoms can be `Vec<char>` or `Vec<&str>` or ...?) or
//! when working with numeric tries.
//!
//! Since the most common use of a tries is to store the chars of a String,
//! a convenience type, [`crate::trie::TrieString`] is provided. The second
//! most common use is to hold a Vec of atoms, for which the
//! [`crate::trie::TrieVec`] type is provided.
//!
//! If these types don't suffice, then you must use the [`crate::trie::Trie`]
//! type directly.
//!
//!
//! Examples:
//! * trie : [`crate::trie`]
//! * iterator : [`crate::iterator`]
//!
//! Typical usages for this data structure:
//!  - Interning
//!  - Storing large numbers of keys with significant amounts of
//!    sub-key duplication
//!  - Prefix matching keys
//!  - ...

#[cfg(feature = "serde")]
extern crate serde_crate;

pub mod iterator;

pub mod trie;
