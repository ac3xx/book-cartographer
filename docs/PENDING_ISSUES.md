# Pending Issues in EPUB Character Graph Generator

## High Priority

✅ **Fixed:** ~~**1. Missing Character Interaction Counts**~~

~~**Problem**: The relationship extraction process doesn't populate interaction counts, which are critical for accurate centrality calculations.~~

**Solution Implemented**:
- Updated relationship extraction prompt to explicitly request interaction metrics
- Added explicit instructions to quantify interaction frequency and relationship strength
- Implemented character co-occurrence detection to estimate interaction counts 
- Added fallback logic to estimate interaction counts based on relationship strength
- All relationship metrics now include non-zero interaction_count and mention_together_count values
- Character metrics now include interaction_counts for all character connections

## Medium Priority

✅ **Fixed:** ~~**2. Pydantic Deprecation Warnings**~~

~~**Problem**: Multiple uses of deprecated `.dict()` method in Pydantic models.~~

**Solution Implemented**:
- Updated all `.dict()` calls to use `.model_dump()` in character_graph.py
- Added fallback support for both Pydantic v1 and v2 in llm_processor.py
- Maintains backward compatibility while fixing deprecation warnings

### 3. Test Failure in Entity Extractor

**Problem**: The `test_extract_all_entities` test is failing.

**Details**:
- Error: "TypeError: 'coroutine' object is not subscriptable"
- Likely needs to use `await` for an async function call
- Test doesn't handle the asynchronous nature of the entity extractor

### 4. Limited Graph Connectivity

**Problem**: Character graphs often have too few relationships to enable meaningful network analysis.

**Impact**: Centrality scores become less useful with sparse graphs.

**Details**:
- Dungeon Crawler Carl graph had only 3 relationships
- No mechanism to detect implicit relationships

**Potential Solution**:
- Detect character co-occurrence in scenes/chapters
- Add relationship inference based on shared contexts
- Implement a minimum relationship threshold per character

## Low Priority

### 5. Hard-coded Thresholds

**Problem**: Character type classification thresholds are hard-coded.

**Details**:
- Lines 483-491 in `character_graph.py` use fixed thresholds
- Major character threshold: importance > 6.5
- Supporting character threshold: importance > 4.0
- These should adapt to the book size and character count

### 6. Limited Test Coverage

**Problem**: Tests don't verify behavior with actual book data.

**Details**:
- Current tests use simplified relationship structures
- No end-to-end tests with real EPUB files
- Test data doesn't include edge cases (disconnected graphs, etc.)

### 7. Incomplete Real-world Testing

**Problem**: Haven't tested centrality calculation with the actual EPUB processing pipeline.

**Details**:
- Only tested with manually-created relationship data
- Need to test the entire extraction and analysis pipeline

### 8. Runtime Performance

**Problem**: Centrality algorithms may be slow for very large character graphs.

**Details**:
- Current implementation tries multiple algorithms if one fails
- No performance benchmarks for large books with many characters
- Could become an issue with very complex books

## Completed Issues

### ✅ Weighted Centrality Calculation

**Solution**: Implemented in `character_graph.py`
- Uses interaction counts as edge weights
- Falls back to more robust algorithms if needed
- Includes mention counts in importance calculation
- Produces more differentiated and accurate centrality scores

**Documentation**: See `CENTRALITY_FIX.md`