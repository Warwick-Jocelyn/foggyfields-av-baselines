# Dataset clip notes — lighting / weather / naming (READ BEFORE COMPARING CLIPS)

The clip folder names encode scene attributes. **Some clips are captured at NIGHT** —
this matters when interpreting/comparing per-clip metrics, because lighting differs.

## NVIDIA_AV_Fog (the fog benchmark)
All benchmark clips are **fog + DAYTIME** (folder names `..._fog_day_...`):

| Clip | Folder | Weather | Lighting | Scene |
|---|---|---|---|---|
| 002 | `002_..._fog_day_rural` | fog (medium) | day | rural, ~113 km/h |
| 003 | `003_..._fog_day_residual` | fog (light) | day | residential |
| 004 | `004_..._fog_day_highway_with_cars` | fog (heavy) | day | highway |
| 006 | `006_..._fog_day_residence` | fog | day | residence (available, not in benchmark) |
| 008 | `008_..._fog_day_rural` | fog | day | rural (available, not in benchmark) |

→ No nighttime clips in the NVIDIA_AV_Fog set used here.

## PandaSet (fog-free reference) — ⚠️ contains NIGHT clips
The PandaSet clip folders were **renamed to mark lighting/scene** (e.g. `_night`, `_night_glare`).
Our two benchmark clips differ in lighting:

| Clip | Folder | Lighting | Note |
|---|---|---|---|
| **011** | `011-1` | **day** | daytime reference |
| **078** | `078_night_1` | **NIGHT** | nighttime — higher PSNR partly reflects the night scene, **not** directly comparable to 011 |

**Full list of NIGHTTIME PandaSet clips** (renamed locally): `057_night`, `059_night_difficult`,
`062_night_padestrains`, `068_night_glare`, `070_night_galre_noisy`, `072_night`, `073_night`,
`074_night_padestrains`, `077_night`, `078_night_1`, `149_night`.

## Why this reminder exists
PandaSet is fog-free, so it serves as the "no-fog" baseline that isolates the fog-specific
behaviour seen on NVIDIA_AV_Fog. But **clip 078 is a night scene** — keep that in mind:
- when comparing 011 (day) vs 078 (night) numbers, lighting is a confound;
- if you later add more PandaSet clips, check the folder name for a `_night` tag first.

> The processed datasets themselves are **not** in this repo or on HuggingFace
> (NVIDIA_AV_Fog stays local / regenerable from NVIDIA PhysicalAI-AV; PandaSet is public).
> Only this naming/lighting metadata is tracked here as a reminder.
