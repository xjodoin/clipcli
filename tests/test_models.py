from clipcli.models import Transcript, TranscriptSegment, format_seconds, parse_timestamp


def test_parse_timestamp_variants() -> None:
    assert parse_timestamp("12") == 12
    assert parse_timestamp("01:02") == 62
    assert parse_timestamp("01:02:03") == 3723


def test_transcript_prompt_lines_are_timestamped() -> None:
    transcript = Transcript(
        segments=[
            TranscriptSegment(text="A strong opening line.", start=0, end=4.2, speaker="S1"),
            TranscriptSegment(text="The payoff.", start=4.2, end=9),
        ]
    )

    lines = transcript.as_prompt_lines()

    assert "[00:00 - 00:04] S1: A strong opening line." in lines
    assert "[00:04 - 00:09] The payoff." in lines


def test_format_seconds() -> None:
    assert format_seconds(62) == "01:02"
    assert format_seconds(3723) == "01:02:03"
