# Cozmo AI Case Study: Phone Scan → Floor Plan (basic prototype)

## What this is, in plain words

You walk around your home holding an iPhone that has a LiDAR sensor (Pro models). LiDAR is a small laser that measures the distance to everything the camera sees. A free app records those distances plus where the phone was at each moment.

This project turns that recording into a **floor plan**: a top-down drawing of every room, like the plans made by Polycam or magicplan. The plan shows:

- each room's **shape** and **floor area**,
- the **length of every wall**,
- the **ceiling height**,
- **doors and windows** with their widths, and which rooms each door connects.

Every number comes with a **range** (for example `3.06 m, likely between 3.04 and 3.08`), so you can see how sure the system is.

## How it works (simple version)

1. **Collect points.** Each LiDAR frame becomes thousands of 3D dots on the walls, floor and ceiling. All frames are merged into one 3D "dot cloud" of the home.
2. **Find up and down.** The lowest large flat surface is the floor. The highest one facing down is the ceiling. Their difference is the ceiling height.
3. **Split into rooms.** Above every door there is a strip of wall (the header) that hangs below the ceiling, so the ceiling breaks at each doorway. Each unbroken piece of ceiling is one room.
4. **Find the walls.** Walls are where lots of dots line up on a vertical plane. Using thousands of dots per wall gives centimetre-level positions.
5. **Find doors and windows.** A gap in a wall that the scanner saw through is a door. A gap above a solid sill is a window.
6. **Draw it.** Output is `result.json` (all numbers) and `plan.png` (the picture).

## Quick start (about 5 minutes)

```bash
# 1. Python 3.10+ environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. One command per capture
python scan2plan.py data/c7d28f72c6          # -> out/c7d28f72c6/result.json + plan.png
```

## How to capture (for anyone, no engineering needed)

1. Use an iPhone **Pro** (12 Pro or newer; we used a 14 Pro). Install **Stray Scanner** from the App Store (free).
2. Turn on all the lights and open every interior door fully.
3. Press record. Hold the phone at chest height and walk slowly (about half a step per second) along the walls of each room.
4. Tilt the phone **up to the ceiling** and **down to the floor** once in every room. The ceiling is needed to measure ceiling height and to split rooms.
5. Walk through each doorway slowly, pointing the phone through it.
6. Avoid pointing at mirrors and large glass for long, and don't run.
7. Stop recording. In the app, export the recording (Files → AirDrop or USB) and copy the folder into `data/`.

## Results on our three scans (iPhone 14 Pro, Stray Scanner)

| Capture | What it is | Result |
|---|---|---|
| `c7d28f72c6` | whole flat, ceiling scanned | 5 rooms, 49.0 m², ceilings 2.34–3.08 m |
| `1a8384c3f6` | floor-only scan (no ceiling) | 1 merged area, 41.8 m² (no ceiling, so rooms can't be split) |
| `c00a170fe1` | single room, no ceiling | 11.8 m² |

Accuracy can't be stated yet: no tape-measure ground truth has been taken for these rooms. See `todo.md`.

## Known limits of this prototype

- **LiDAR tier only.** Photo and video tiers are not implemented yet.
- Rooms are assumed to have **square corners**; angled walls will be squared off.
- If the **ceiling isn't scanned**, rooms joined by wide openings merge into one.
- Some false windows or doors appear where large furniture hides a wall.
- No drift correction yet. ARKit poses are used as they are.

## Files

| File | Purpose |
|---|---|
| `scan2plan.py` | the whole pipeline (one script) |
| `requirements.txt` | Python packages |
| `todo.md` | progress checklist |
| `task_summary.md` | short summary of the task and what was delivered |
| `data/` | raw scans (not in git: too large) |
| `out/` | generated results (not in git; re-create with the command above) |
