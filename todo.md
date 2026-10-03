# Cozmo AI Case Study — TODO

Legend: `[x]` done · `[ ]` not started
Owner: **C** = code · **YOU** = needs the phone, the rooms and a tape/laser

> Hardware: **iPhone 14 Pro (LiDAR)** → all three tiers can be captured on one phone.
> Data in `data/` (Stray Scanner, LiDAR tier): `c00a170fe1` single room · `1a8384c3f6` floor-only scan ·
> `c7d28f72c6` whole flat with ceiling. The `.zip` files are identical copies. Not in git (1.7 GB).

## Basic prototype (done)
- [x] C — Git repo, Python environment, `requirements.txt`
- [x] C — Single script `scan2plan.py`: Stray Scanner capture → `result.json` + `plan.png`
- [x] C — Floor / ceiling height, room split (ceiling headers), walls, doors/windows, door→room links
- [x] C — 90 % range on every measurement (sensor prior, not yet calibrated)
- [x] C — Ran on all 3 real captures
- [x] C — README (plain language), task_summary.md

## Next — accuracy proof (most valuable)
- [ ] YOU — Tape/laser-measure the rooms in `c7d28f72c6`: every wall, door, window, ceiling height
- [ ] C — Ground-truth file + scorer: error per wall / door / ceiling against the gates
- [ ] YOU — Re-scan one room twice (repeatability gate)

## Full brief — remaining
- [ ] C — Photo tier (2–8 photos per room folder, pretrained depth model) + stitching by door matching
- [ ] C — Video tier (native camera clip)
- [ ] C — Drift correction + on/off ablation on the multi-room scan
- [ ] C — Damage detection (stains, cracks) + concealed-damage rules + repair scope
- [ ] YOU — Polycam (free) scans of 2 rooms → head-to-head table
- [ ] C+YOU — Fix loop: worst gate → root cause → fix → before/after
- [ ] C — One-page capture protocol, device matrix, compliance matrix, ≤ 6-page report
