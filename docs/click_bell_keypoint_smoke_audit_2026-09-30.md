# ClickBell keypoint smoke audit — 2026-09-30

## Scope

This audit covers the one-episode dataset collected from source commit
`9c1b210` on a Secure RunPod A40. It evaluates label integrity only; it does
not measure policy performance.

The dataset contains one successful 74-frame expert episode, 74 keypoint rows,
and one video for each of `cam_high`, `cam_right_wrist`, and
`cam_left_wrist`. The frame and label counts match exactly.

## Visual findings

- `cam_high` sees the button in all 74 frames. Its mask centroid lands near the
  circular button center before contact.
- The high-camera X label shifts from `0.7487` to `0.7339`, or 9.4 pixels at
  640-pixel width, while the fixed bell is partially occluded by the right
  gripper. A visible-mask centroid is therefore not an occlusion-invariant
  physical target.
- `cam_right_wrist` sees the button in frames 14–73. At first visibility the
  centroid is near the bottom image boundary (`y=0.997`), where only a small
  part of the button is visible. A binary any-pixel visibility label treats
  this weak observation as fully valid.
- `cam_left_wrist` sees no button pixels in any frame. It should not contribute
  a bell-localization loss in the first probe.
- Appearance changes are substantial, but the visible centroid generally
  remains on the circular button. This is promising for an auxiliary geometry
  target, provided occlusion and truncation are handled explicitly.

## Required label changes before scaling collection

1. Record mask area and bounding-box truncation, then supervise a confidence or
   valid-label mask rather than treating any visible pixel as equally reliable.
2. Add an occlusion-invariant target by projecting the known press-point pose
   into each camera. Keep mask-derived visibility as a separate field.
3. Add the right tool-tip projection and relative tool-to-press-point offset.
   Contact depends on relative geometry, not bell position alone.
4. Start the auxiliary loss with `cam_high` and `cam_right_wrist`; exclude
   `cam_left_wrist` unless a later audit shows useful coverage.
5. Re-run a one-episode smoke and overlay the projected point, mask centroid,
   and tool-tip together before collecting the nine-cell pilot.

## Implemented follow-up

`scripts/collect_click_bell_keypoint_demos.py` now writes sidecar schema v2.
The original `keypoints` array remains unchanged for the existing ACT loader,
and each high/right-wrist camera row additionally records:

- visible-mask centroid, pixel area, fractional area, bounding box, and whether
  the box touches an image boundary;
- the projected physical center of the moving button-cover surface;
- the projected CobotMagic right-arm TCP and its image-plane displacement to
  the press point;
- in-front/in-frame flags, whether the projected press point is represented in
  the visible button mask, and a conservative centroid-confidence weight.

The physical offsets are derived from the pinned simulator sources: the button
cover mesh reaches 0.02904 m along its local Z axis, and the CobotMagic OPW
solver defines the TCP 0.143 m along `right_link6` local Z. Projection queries
each live camera's randomized pose and intrinsics on every frame.

Unit coverage verifies the DexSim OpenGL camera conversion, off-screen and
behind-camera flags, mask truncation, rasterized press-point lookup, schema
serialization, and coexistence with the legacy labels. The following live
smoke validates those source-derived conventions against rendered pixels.

## Schema-v2 live validation — 2026-10-01

Commit `516ea4e` completed a second one-episode Secure A40 smoke. The resulting
dataset has 74 frames, 74 strict-JSON sidecar rows, schema version 2, and three
complete videos. The source-derived projections agree with the rendered robot
and bell geometry:

- `cam_high`: the press point is in-frame for 74/74 frames, the tool tip for
  55/74, and 63/74 mask-centroid labels receive confidence 1. The other 11 are
  rejected when the press-point neighborhood becomes occluded.
- `cam_right_wrist`: the visible button mask exists in 60/74 frames, touches an
  image boundary in 25, the physical press point is in-frame in 54, and the
  tool tip is in-frame in all 74. The conservative confidence accepts 32/74.
- The old whole-articulation mask centroid differs from the projected physical
  point by a median 8.3 pixels in the high camera and 27.6 pixels in the moving
  wrist camera. The overlays show why the physical point is preferable: the
  wrist view has strong perspective, truncation, and contact occlusion while
  the projected point remains tied to the moving cover surface.

This validates collection and projection conventions for a small multi-cell
pilot. It does not yet validate an auxiliary-loss formulation or policy gain.

## Artifacts

- Full overlays: `artifacts/keypoint_smoke/20260930/overlays/`
- Contact-window sheets: `artifacts/keypoint_smoke/20260930/contact_window/`
- Reusable renderer: `scripts/visualize_click_bell_keypoints.py`
- Schema-v2 archive:
  `artifacts/keypoint_smoke/20261001/keypoint_geometry_smoke_516ea4e.tar.gz`
  (SHA256 `c1f27587e367370edf60bc326ad8329a9b000cc666c5d7efce062d435ba6140b`)
