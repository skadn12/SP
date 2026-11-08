"""Extra completion gate over the frozen provenance validator, no side effects."""
from production_core import read, require


def validate_completion(result, events):
    terminal = read(result["terminal_binding"]["path"])
    for field in ("returncode", "timed_out"):
        require(type(result.get(field)) is type(terminal.get(field)) and result.get(field) == terminal.get(field),
                "RESULT_TERMINAL_COMPLETION_MISMATCH", field=field)
    require(type(result.get("returncode")) is int and result["returncode"] == 0,
            "BACKEND_NONZERO_OR_UNKNOWN_EXIT", returncode=result.get("returncode"))
    require(result.get("timed_out") is False, "BACKEND_TIMED_OUT_OR_UNKNOWN")
    require(events.get("turn_completed") is True, "BACKEND_TURN_NOT_COMPLETED")
    return {"status": "PASS_BACKEND_NORMAL_COMPLETION", "returncode": 0, "timed_out": False,
            "turn_completed": True, "thread_id": events.get("thread_id")}
