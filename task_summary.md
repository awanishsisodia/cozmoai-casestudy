# Task Summary

## The task (Cozmo AI case study)

Build a system that turns a phone capture of a home into a **measured floor plan**, accurate enough to compete with consumer apps like Polycam or magicplan.

The full brief asks for:

- **Three input tiers:** plain photos, a video walkthrough, and LiDAR scans.
- **Per-room output:** walls, ceiling height, floor area, doors and windows.
- **One stitched whole-home plan.**
- **Damage detection:** stains and cracks, plus repair line items.
- **A confidence range on every number.**
- **A self-built benchmark with tape-measure ground truth,** scored against strict accuracy gates (for example, door widths within 2 cm).
- **A head-to-head against a consumer app.**
- **A "fix loop":** find your worst result, fix it, and prove the improvement.
- **Real commit history.**

## What was delivered (basic prototype, time-boxed)

The scope was deliberately cut to a **working, accurate LiDAR-tier prototype in a single script**.

- **`scan2plan.py`** (about 300 lines) takes one Stray Scanner capture and produces `result.json` + `plan.png`. Each run takes seconds.
- **Floor and ceiling:** median of thousands of flat-surface points, giving centimetre precision.
- **Room splitting:** uses the break in the ceiling at door headers. When the ceiling wasn't scanned, it falls back to cutting the floor map at doorways.
- **Walls:** 1 cm histograms of wall points. Walls are axis-aligned to the dominant wall direction.
- **Doors and windows:** wall gaps the scanner saw through (doors), or gaps above a solid sill (windows). Doors are linked to the room on the other side.
- **Ranges:** every measurement has a 90% range. These come from a LiDAR sensor prior and are **not yet calibrated** against ground truth.
- **Tested on 3 real iPhone 14 Pro scans.** The full flat scan gives 5 rooms, 49 m², and ceiling heights of 2.34–3.08 m.

### Key finding during the build

Stray Scanner's poses already use OpenCV camera axes, not the ARKit convention. I found this from the data itself: with the ARKit flip, the floor ended up 18 cm below the phone; without it, 1.4 m, which is correct.

## What remains for the full brief

See `todo.md`. In short:

- **Ground truth:** tape-measure the scanned rooms.
- **Photo and video tiers.**
- **Drift correction** with an on/off comparison.
- **Damage detection** and repair scope.
- **Benchmark scorer.**
- **Polycam head-to-head.**
- **Fix loop.**
- **Compliance matrix.**
