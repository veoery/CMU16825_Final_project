# Evaluation Report: Entity Sequence Analysis

## Data Row Parsing

**File:** `00900284_00001_repaired_20251201_0131.json`

| Metric | Value | Description |
|--------|-------|-------------|
| status | success ✓ | Successfully loaded and compared |
| entity_count_acc | 1.0 | Entity count accuracy |
| type_seq_acc | 0.667 | Entity type sequence accuracy |
| type_dist_sim | 0.5 | Entity type distribution similarity |
| gen_count | 3 | Number of generated entities |
| gt_count | 3 | Number of ground truth entities |

---

## Metric 1: Entity Count Accuracy = 1.0 (100%)

### Meaning
Matching degree between generated entity count and ground truth entity count.

### Formula
```
1.0 - |gen_count - gt_count| / gt_count
    = 1.0 - |3 - 3| / 3
    = 1.0 - 0
    = 1.0 ✓ Perfect match
```

### Interpretation
You generated 3 entities, and the ground truth also has 3 entities. This is completely correct!

**✓ Generated the correct number of entities**

---

## Metric 2: Type Sequence Accuracy = 0.667 (~67%)

### Meaning
Matching degree between generated entity type sequence and ground truth sequence.
Uses prefix matching: number of consecutive matches / total count

### Comparison

**Your generated sequence:**
```
[Sketch, ExtrudeFeature, Sketch]
           ↑ Match
```

**Ground truth sequence:**
```
[Sketch, ExtrudeFeature, ExtrudeFeature]
↑ Match  ↑ Match      ↓ Mismatch!
```

**Matches:** 2 / 3 = 0.667 ✓

### Interpretation
- **✓** 1st entity type correct (Sketch)
- **✓** 2nd entity type correct (ExtrudeFeature)
- **✗** 3rd entity type incorrect (generated Sketch, ground truth is ExtrudeFeature)

### Problem
Sequence order is not completely correct; the 3rd feature type is wrong.

---

## Metric 3: Type Distribution Similarity = 0.5 (50%)

### Meaning
Similarity of entity type distribution (regardless of order).
Uses Jaccard similarity coefficient.

### Comparison

**Your generated distribution:**
```
{Sketch: 2, ExtrudeFeature: 1}
```

**Ground truth distribution:**
```
{Sketch: 1, ExtrudeFeature: 2}
```

### Jaccard Calculation
```
Jaccard = Intersection size / Union size

Intersection: min(2,1) + min(1,2) = 1 + 1 = 2
Union:        max(2,1) + max(1,2) = 2 + 2 = 4
Similarity:   2/4 = 0.5 ✓
```

### Interpretation
- **✗** You generated 2 Sketch entities, ground truth only has 1
- **✗** You generated 1 ExtrudeFeature, ground truth has 2
- **→** Type distribution is incorrect

---

## Overall Assessment

### Your Generation Results

| Aspect | Score | Level | Notes |
|--------|-------|-------|-------|
| Count | 1.0 | ✓ Perfect | — |
| Sequence | 0.667 | ⚠ Moderate | 3rd entity incorrect |
| Distribution | 0.5 | ⚠ Fair | Wrong ratios |

### Problem Diagnosis

1. **✓** Generated correct number of entities
2. **✓** First two features are correct
3. **✗** Third feature type is incorrect
4. **✗** Feature proportion is wrong (too many Sketch, insufficient ExtrudeFeature)

### Recommendations

- Model learning is reasonable, but prediction accuracy decreases for 3rd+ features
- May need to improve long sequence prediction capability
- Training data may have few samples of length-3, leading to insufficient generalization

---

## How to Improve Scores

### type_seq_acc: 0.667 → 1.0
Requires correct prediction of the 3rd feature as well.

### type_dist_sim: 0.5 → 1.0
Requires correct quantity for each entity type:
- Change Sketch from 2 to 1
- Change ExtrudeFeature from 1 to 2

### entity_count_acc: Keep at 1.0
Continue maintaining current entity count prediction ability.