import json

from clipcli.gemini import parse_plan_json


def test_parse_plan_json_normalizes_supoclip_style_fields() -> None:
    raw = {
        "summary": "A useful interview.",
        "key_topics": ["growth"],
        "most_relevant_segments": [
            {
                "title": "Growth lesson",
                "start_time": "00:10",
                "end_time": "00:45",
                "text": "You need a stronger hook.",
                "reasoning": "Complete thought.",
                "hashtags": "growth clips",
                "virality": {
                    "hook_score": 20,
                    "engagement_score": 18,
                    "value_score": 22,
                    "shareability_score": 17,
                    "total_score": 77,
                    "hook_type": "statement",
                    "virality_reasoning": "Direct and useful.",
                },
                "broll_opportunities": [
                    {
                        "timestamp": "00:12",
                        "duration": 3,
                        "search_term": "founder reviewing analytics dashboard",
                    }
                ],
            }
        ],
    }

    result = parse_plan_json(f"```json\n{json.dumps(raw)}\n```")

    assert result.clips[0].start == 10
    assert result.topics == ["growth"]
    assert result.clips[0].end == 45
    assert result.clips[0].hashtags == ["#growth", "#clips"]
    assert result.clips[0].virality.reasoning == "Direct and useful."
    assert result.clips[0].broll[0].prompt == "founder reviewing analytics dashboard"


def test_parse_plan_json_filters_duration() -> None:
    raw = {
        "summary": "",
        "clips": [
            {
                "title": "Too short",
                "start": 0,
                "end": 4,
                "summary": "x",
                "hook": "x",
                "caption": "x",
            },
            {
                "title": "Good",
                "start": 0,
                "end": 20,
                "summary": "x",
                "hook": "x",
                "caption": "x",
            },
        ],
    }

    result = parse_plan_json(json.dumps(raw), min_seconds=15, max_seconds=60)

    assert [clip.title for clip in result.clips] == ["Good"]


def test_parse_plan_json_removes_candidates_if_filter_removes_all() -> None:
    raw = {
        "summary": "",
        "clips": [
            {
                "title": "Short but usable",
                "start": 0,
                "end": 4,
                "summary": "x",
                "hook": "x",
                "caption": "x",
            }
        ],
    }

    result = parse_plan_json(json.dumps(raw), min_seconds=15, max_seconds=60)

    assert result.clips == []


def test_parse_plan_json_accepts_list_payload() -> None:
    result = parse_plan_json(
        json.dumps(
            [
                {
                    "title": "List Clip",
                    "start_time": "00:01",
                    "end_time": "00:21",
                    "summary": "x",
                    "hook": "x",
                    "caption": "x",
                }
            ]
        )
    )

    assert result.clips[0].title == "List Clip"
    assert result.clips[0].start == 1
