# Freestyle metric spec (v1)

Plan task 0.1. One entry per metric. Yan signs this off in task 0.2: add, remove or rename metrics, and mark the five that matter most.

Every metric is a pure function over the 13 keypoints at one sampled instant, plus a small context. Separating **what is measured** (the formula) from **when it is sampled** (the instant) is what lets the same formula serve other strokes later.

## Keypoints

Index order is fixed in `swimtech/types.py`:

| idx | name | idx | name |
|---|---|---|---|
| 0 | nose | 7 | L wrist |
| 1 | L ear | 8 | R wrist |
| 2 | R ear | 9 | L hip |
| 3 | L shoulder | 10 | R hip |
| 4 | R shoulder | 11 | L knee |
| 5 | L elbow | 12 | R knee |
| 6 | R elbow | | |

Derived points used below:

- **shoulder line**: the line through L and R shoulder.
- **shoulder mid**, **hip mid**, **ear mid**: midpoints of the pairs.
- **body axis**: hip mid → shoulder mid.
- **shoulder width ref**: the median L–R shoulder distance over the current length. Used to normalize body-relative distances so the near and far ends of the pool compare.
- **lane dir**: unit direction of travel for the current length, from the track's x reversal (task 2.3).

## Instants (from task 2.3)

Per arm, per cycle:

- **entry**: the frame the wrist enters the water in front of the shoulder.
- **midrec**: the frame the wrist is highest above the shoulder line during recovery.
- **exit**: the frame the wrist leaves the water near the hip.
- **breath**: the frame of maximum head yaw on a breathing cycle, or none.

Rule for the oblique camera (open issue 1): all "vertical" signals are measured **perpendicular to the shoulder line**, with positive on the side away from the hips. Never raw image y.

## Spaces

- **image**: pixel coordinates of the normalized video. Angles are fine here because the recovery arm is above the water and not on the lane plane anyway.
- **rect**: after lane-plane homography (task 1.5), in meters. Only near-surface points (shoulders, hips, wrist at entry) are valid here. Metric names ending in `_rect` use it. When no calibration exists the `_rect` metrics are omitted and the results JSON carries `"calibrated": false` (open issue 7).

## Angle metrics

Interior angle at B from A, B, C: `degrees(acos(dot(A−B, C−B) / (|A−B||C−B|)))`.

| name | per arm | instant | keypoints | formula | unit | space |
|---|---|---|---|---|---|---|
| `elbow_angle_exit` | L/R | exit | shoulder, elbow, wrist | interior angle at elbow | deg | image |
| `elbow_angle_midrec` | L/R | midrec | shoulder, elbow, wrist | interior angle at elbow | deg | image |
| `elbow_angle_entry` | L/R | entry | shoulder, elbow, wrist | interior angle at elbow | deg | image |
| `forearm_entry_angle` | L/R | entry | elbow, wrist, shoulders, hips | angle between (wrist − elbow) and body axis | deg | image |
| `head_angle` | – | midrec, non-breath cycles only | nose, ears, shoulders | line angle between (nose − ear mid) and shoulder line, 0–90 | deg | image |
| `shoulder_line_angle` | – | entry | shoulders | line angle between shoulder line and lane dir, 0–90; roll proxy | deg | image |

## Distance metrics

| name | per arm | instant | keypoints | formula | unit | space |
|---|---|---|---|---|---|---|
| `wrist_height_midrec` | L/R | midrec | wrist, shoulders, hips | perpendicular distance of wrist from shoulder line (positive away from hips) ÷ shoulder width ref | sw | image |
| `head_lift_breath` | – | breath | nose, shoulders, hips | nose height above shoulder line on the breath frame − median nose height at midrec on non-breath cycles this length, ÷ shoulder width ref | sw | image |
| `shoulder_foreshortening` | – | entry | shoulders | shoulder distance ÷ shoulder width ref; < 1 means rolled | ratio | image |
| `hand_entry_offset_rect` | L/R | entry | wrist, shoulders, hips | signed lateral distance of wrist from the body axis line; positive = across the midline | m | rect |
| `hip_shoulder_offset_rect` | – | entry | shoulders, hips | signed lateral distance of hip mid from the line through shoulder mid along lane dir | m | rect |

"sw" = shoulder widths.

## Rhythm and breathing metrics

Computed from the cycles array, not sampled at a frame. Reported per length and per race.

| name | formula | unit |
|---|---|---|
| `breath_side` | per cycle: −1 if head yaw proxy says left, +1 right, 0 none | side |
| `breath_frequency` | breaths ÷ cycles over a length | per_cycle |
| `breath_cycle_penalty` | mean duration of breath cycles − mean duration of non-breath cycles, same length | s |
| `stroke_rate` | (L cycles + R cycles) ÷ length duration × 60 | spm |
| `cycle_cv` | std ÷ mean of cycle durations over a length, both arms | ratio |

Head yaw proxy (task 2.3): ratio of (nose − L ear) to (nose − R ear) distance, log-scaled; a breath is ≥ 4 consecutive frames beyond a threshold.

## Aggregation (task 2.4)

- Per cycle → per length → per race. Per-length values are medians over cycles; per-race values are medians over lengths.
- **Asymmetry** for each per-arm metric: `L − R`, signed, computed **only within lengths swum in the same direction** (open issue 2), then averaged. Reported as `<name>_asym`.
- **Fatigue drift**: ordinary least-squares slope of the per-cycle value against cycle index across the race, reported as `<name>_slope` in `unit/cycle`.
- **Baseline delta**: once per-swimmer baselines exist (task 7.3), every per-race value also carries `delta` from the swimmer's rolling median. The UI leads with the delta.

## Visibility

Every value carries `vis` = mean of (keypoint vis ÷ 2) over the keypoints the metric used, at the sampled frame. Aggregates carry the mean `vis` of their inputs. The UI greys out anything under 0.6. A metric whose inputs are all NaN at its instant is omitted for that cycle, never reported as 0.

## Sign-off

Yan: edit this table directly or reply in chat. Mark the five most important metrics with a ★ in the name column.
