# Centrality Calculation Improvements

## Problem

When processing "Dungeon Crawler Carl," the centrality scores didn't accurately reflect character importance:

- Carl, Donut, and Bea all had identical scores (0.577)
- All other characters had nearly zero scores (3.62e-07)

This happened because:
1. The graph contained a triangle/clique of three characters with no weights
2. Eigenvector centrality gives identical scores to nodes in a perfect clique
3. The graph had disconnected components

## Solution

We improved the centrality calculation by:

1. Using interaction counts and mention counts as edge weights
2. Implementing a more robust centrality calculation algorithm with fallbacks:
   - First try eigenvector_centrality_numpy with weights (most accurate)
   - Fall back to standard eigenvector_centrality with increased iterations
   - Fall back to PageRank if eigenvector centrality fails
   - Use degree centrality as a last resort
3. Incorporating mention counts as a factor in the importance score calculation
4. Normalizing mention counts relative to the highest mentioned character

## Results

After implementing these changes, we get much more realistic centrality scores:

### Before:
```
Carl: 0.577350
Donut: 0.577350
Bea: 0.577350
All others: ~0
```

### After:
```
Carl: 0.685974
Donut: 0.670360
Bea: 0.248202
Mongo: 0.111905
Mordecai: 0.058469
Samantha: 0.050116
```

The importance scores are even more differentiated, properly reflecting the actual importance of characters in the story.

## Implementation Details

1. Created a weighted copy of the graph using interaction_count and mention_together_count
2. Modified eigenvector centrality calculation to use these weights
3. Added fallback mechanisms to handle convergence issues
4. Incorporated mention_count from character metrics in the importance calculation
5. Created a more balanced importance formula that considers:
   - Weighted eigenvector centrality (40%)
   - Betweenness centrality (20%)
   - Degree centrality (20%)
   - Mention count factor (20%)

## Files Changed

- `/epub_character_graph/character_graph.py`: Updated calculate_centrality_metrics method

## Testing

Added a new test case `test_weighted_centrality_calculation` that verifies:
1. Carl and Donut have higher centrality than Bea
2. Carl has the highest importance score
3. Minor characters have appropriate lower scores
4. All metrics are properly calculated and stored