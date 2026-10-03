# Cozmo AI Case Study — TODO

Legend: `[x]` done · `[~]` code done, needs real-world data from you · `[ ]` not started
Owner: **C** = can be done in code (Claude/you at the keyboard) · **YOU** = needs a phone, a room and a tape/laser

> Hardware note: an **iPhone 15 Plus has no LiDAR**. It covers the Photo and Video tiers.
> The LiDAR tier needs an iPhone 12 Pro or newer **Pro** model (borrow one for one afternoon).

## 0. Setup
- [ ] C — Project folder, git repo, first commit
- [ ] C — Python environment (`uv`), dependency list, one-line install
- [ ] C — Script that downloads model weights (no weights in git)

## 1. Capture route (Part 1) — Route 2: stock capture protocol
- [ ] C — Pick stock apps: native Camera (photos, video), Stray Scanner (LiDAR: depth + poses + intrinsics)
- [ ] C — One-page capture protocol a non-engineer can follow (`docs/capture_protocol.md`)
- [ ] C — Device matrix: tier → hardware → honest accuracy (`docs/device_matrix.md`)
- [ ] C — Mirrors / glass / wet surfaces / low-light guidance + pipeline handling

## 2. Pipeline (Part 2) — one command per capture
- [ ] C — Output JSON schema (`schema/capture_output.schema.json`)
- [ ] C — Loaders: Stray Scanner folder, video file, per-room photo folders → common "posed RGB-D frames"
- [ ] C — Photo/video front-end: metric monocular depth model + feature-based pose estimation
- [ ] C — Geometry back-end: floor/ceiling/wall planes, room polygon, ceiling height, floor area
- [ ] C — Openings (doors/windows) detection + widths
- [ ] C — Multi-room segmentation (continuous capture) and stitching (per-room folders)
- [ ] C — Drift correction (plane-anchored + Manhattan yaw) with on/off switch for the ablation
- [ ] C — Damage detection (open-vocabulary model) → per-surface regions with class + m²
- [ ] C — Concealed-damage rules (with the rule ID that fired)
- [ ] C — Scope line items keyed to surfaces
- [ ] C — Confidence interval on every measurement, per-tier calibration file
- [ ] C — Rendered stitched floor plan (PNG/SVG)
- [ ] C — `cozmo run <capture>` single command; JSON validated against the schema
- [ ] C — Synthetic test capture so the whole pipeline is tested without a phone

## 3. Benchmark (Part 2 gates) — needs real captures
- [ ] C — Ground-truth template + evaluator that scores every gate
- [ ] YOU — Multi-room capture (≥3 rooms + hallway) at all 3 tiers
- [ ] YOU — Furnished room with staged damage (2 classes, e.g. "water stain" + "crack")
- [ ] YOU — One room captured twice at the same tier (repeatability)
- [ ] YOU — Tape/laser ground truth for every wall, opening, ceiling height
- [ ] YOU — Run `cozmo bench` → benchmark report
- [ ] C — Drift ablation (on vs off) on the multi-room capture

## 4. Head-to-head (Part 3)
- [ ] C — Script that compares our output vs a consumer-app export, dimension by dimension
- [ ] YOU — Scan 2 benchmark rooms with Polycam / magicplan (free tier) and export

## 5. Fix loop (Part 4)
- [ ] C — Before/after run tooling (`scripts/fix_loop.sh`), fix-declaration template
- [ ] YOU+C — Find worst gate on real data → hypothesis → ship fix → before/after

## 6. Process & deliverables (Part 5)
- [ ] C — Commit as we work (real history)
- [ ] C — Compliance matrix (`docs/compliance_matrix.md`)
- [ ] C — README (plain-language + quick start), task_summary.md
- [ ] C — Technical report skeleton (≤ 6 pages)
- [ ] YOU — Raw benchmark data bundle, final report numbers
