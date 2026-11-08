import copy
import json
from pathlib import Path
import unittest

from generate import check_questions, inputs, new_id, parent_id


class RevisionContracts(unittest.TestCase):
    def test_parent_ids_and_one_frozen_request_per_root(self):
        for root in ("136202", "197635", "254852"):
            root_id = f"cc-train-episode-{root}"
            *_, directory, frozen = inputs(root_id)
            request = json.loads((directory / "request.json").read_text())
            self.assertEqual(request["model"], "Qwen/Qwen3.6-35B-A3B")
            self.assertEqual(request["max_tokens"], 12288)
            self.assertEqual(frozen["maximum_http_attempts"], 1)
            self.assertEqual([new_id(root_id, i) for i in range(1, 4)],
                             [f"{root}-qwen36-multi-{i:02d}" for i in (9, 10, 11)])
            self.assertEqual([parent_id(root_id, i) for i in range(1, 4)],
                             [f"{root}-qwen36-multi-{i:02d}" for i in (6, 7, 8)])

    def test_invalid_first_revision_does_not_block_later_revision(self):
        root_id = "cc-train-episode-136202"
        v16, v8, _, _, source, context, prior, *_, directory, _ = inputs(root_id)
        row = json.loads((directory / "PARSED_RESPONSE.json").read_text())["questions"][0]
        later = copy.deepcopy(row)
        later["candidate_id"] = new_id(root_id, 2)
        invalid = copy.deepcopy(row)
        invalid["options"] = ["malformed"]
        invalid["option_evidence"] = [{"turn_ref": []}]
        checks, valid = check_questions(v16, v8, root_id, source, context, prior,
                                        [invalid, later])
        self.assertTrue(checks[0]["errors"])
        self.assertEqual(checks[1]["errors"], [])
        self.assertEqual([q["candidate_id"] for q in valid], [new_id(root_id, 2)])


if __name__ == "__main__":
    unittest.main()
