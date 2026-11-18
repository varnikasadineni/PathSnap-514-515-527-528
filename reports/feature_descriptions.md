# Section 2 - Features (draft by Member A)

One row = one (GPS point, candidate road arc) pair. The classifier answers: "is this arc the one the car is on at this point?"
All features use only the track, nodes and arcs files. The `.route` file is used only to create the 0/1 `label` for training.

| Feature | What it is | Why we use it |
| --- | --- | --- |
| `dist_m` | Shortest distance (m) from the GPS point to the arc | The car is usually close to its road, so the nearest arcs are the likely ones. This is the strongest single signal. |
| `proj_frac` | Where along the arc the point lands (0 = start, 1 = end) | Points that project beyond an end (0 or 1) are probably on the next arc, not this one. Also helps order consecutive points along a road. |
| `arc_len_m` | Length of the arc (m) | Long arcs have a higher chance of containing the point; very short arcs near junctions are easy to confuse. |
| `heading_diff_deg` | Angle (0-180) between the direction the GPS moved and the arc direction | Separates an arc from its reverse twin (same place, opposite direction), which distance alone cannot do. |
| `gps_speed_mps` | Speed estimated from the last GPS step (m/s) | Speed tells how reliable the heading is: when the car is almost stopped, heading is noise. It also separates highway from slow roads. |
| `dt_sec` | Time since the previous GPS point (s) | A big gap means the car may have moved far, so closeness to the previous point means less. Marks irregular sampling. |
| `dist_rank` | 0-based rank of this arc by distance among the point's candidates | Relative closeness: rank 0 is the nearest arc even when all candidates are far (noisy GPS). |
| `dist_gap_to_best_m` | `dist_m` minus the smallest `dist_m` among the point's candidates | How much worse than the best candidate this arc is; small gaps mean an ambiguous point. |
| `cand_count` | Number of candidates found for this point | Many candidates = busy junction or dense network, where the choice is harder. |
| `from_node_deg` | Out-degree of the arc's start node | Tells whether the arc starts at a junction or in the middle of a road. |
| `to_node_deg` | Out-degree of the arc's end node | Same, for the end: dead ends and junctions behave differently. |
| `prev_connected` | 1 if this arc, or an arc leading into it, was a candidate at the previous point | A cheap memory of the previous point, so the model prefers arcs that continue the road the car was on. |

## Notes for the report
- Candidates: arcs within 50 m of the point (widened to 100 / 200 / 350 m if fewer than 5 are found), keeping the 15 nearest. Settings live in `configs/hmm.yaml` so training and prediction match exactly.
- Labels: each GPS point is matched to a route arc with an exact forward-only dynamic programme (`align_track_to_route_monotonic`), so loops and reverse twins cannot steal points.
- The features are saved unscaled; scaling (for Logistic Regression, KNN, MLP) is fitted on training data only inside Member B's cross-validation.
- Very large values exist and are real: `dt_sec` up to 153,064 s (trip 90 has a 42.5 h gap), `gps_speed_mps` up to 180 m/s (GPS jumps). Linear and distance-based models should clip or log-transform these.
