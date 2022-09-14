use trying::trie::TrieVec;
use unicode_segmentation::UnicodeSegmentation;

fn main() {
    // Create our trie
    let mut trie = TrieVec::<&str, usize>::new();

    // Insert some graphemes
    let s = "a̐éö̲\r\n";
    let input = s.graphemes(true);
    let count = input.clone().count();
    trie.insert_with_value(input.clone(), Some(count));
    assert!(trie.contains(input.clone()));
    assert!(trie.get(input.clone()).is_some());
    assert_eq!(trie.get(input), Some(&count));
}
