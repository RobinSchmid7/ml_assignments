# Task 4: Protein Mutation Classification

Classify if mutation is active(1) or inactive(0) depending on mutation information.

The mutation information is encoded in 4 letters, where each letter denotes an amino acid and is
fixed to the corresponding site.

For Example: FCDI
- F = Phenylanine, being in the first site,
- C = Cysteine, being in the second site,
- ...

## Idea

Split mutation information into single characters, ask sklearn about strings as features 
- OneHotEncoder, does not maintain ordering - Update Robin: it maintains the ordering, check the size of the encoded data (try this)!
- Add specific int for specific amino acid (4 dimensions, maintain ordering)
- one more?

*Check if ordering is important or not!*

## Key points

Hyperparameters important for such tasks
- Optimize for f1 score
- Optimized batch size
- etc.
- Start with simple NN e.g. 2 layers, Linear followed by ReLU




