# 9-minute presentation outline + 3-minute Q&A

Aim for 8–9 concise slides, one key point per slide. Everyone should understand the full story even if one person owns each slide.

| Time | Slide | Content / visual |
|---:|---|---|
| 0:00–0:45 | Problem | Face grid → flattened vector; why preserve row/column structure? |
| 0:45–1:30 | Paper claim | MDA and small-sample motivation; cite Yan et al. |
| 1:30–2:35 | Method | DTC intuition and alternating mode update; MDA/2-1 vs 2-2 diagram |
| 2:35–3:25 | Course links | PCA, LDA, generalized eigenvectors, nearest neighbours |
| 3:25–4:20 | Our protocol | Dataset, gallery counts, matched repeated splits, no test-fit leakage |
| 4:20–5:20 | Four methods | PCA / Fisherfaces / MDA-2-1 / MDA-2-2 and fixed dimensions |
| 5:20–6:25 | Results | Accuracy-vs-gallery chart and group mean±SD table; label as group-produced |
| 6:25–7:25 | Critical analysis | Scaling sensitivity, 64×64 vs original, five-repeat uncertainty |
| 7:25–8:30 | Conclusion / extension | Honest partial-reproduction claim; 30+ paired splits and validation-tuned dimensions |
| 8:30–9:00 | Takeaway | One-sentence team insight; transition to Q&A |

## Q&A rehearsal prompts

- Why can feature scaling change 1-NN? Euclidean distances weight axes by their scale.
- Did test data influence projections? No; every projection was fit on training examples only.
- Why do our numbers differ from the article? Resolution, fixed dimensions, split aggregation, and scaling choices differ.
- Does this disprove MDA? No; it is a small, implementation-specific partial reproduction.
- What would improve confidence? More matched splits, paired uncertainty, nested validation, and paper-faithful data/settings.

## Speaker handoff

Agree on slide ownership and time budget in rehearsal, but require every member to answer the Q&A prompts. Attribute paper figures and label generated charts “Group 27 results.”
