from clipcli.autocrop import _center_to_crop_anchor, _sample_offsets


def test_center_to_crop_anchor_for_16x9_left_speaker() -> None:
    anchor = _center_to_crop_anchor(center_x=487, width=1920, height=1080)

    assert 0.12 < anchor < 0.16


def test_center_to_crop_anchor_clamps_edges() -> None:
    assert _center_to_crop_anchor(center_x=0, width=1920, height=1080) == 0
    assert _center_to_crop_anchor(center_x=1920, width=1920, height=1080) == 1


def test_sample_offsets_focus_on_clip_start() -> None:
    offsets = _sample_offsets(duration=20, samples=5)

    assert offsets[0] <= 1.5
    assert offsets[-1] <= 7
