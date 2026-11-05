import json
import os
import unittest
from unittest.mock import patch

from arc_decision.decision_client import (
    LayaLocalClient,
    LocalSystemOneClient,
    LocalSystemOneConfig,
    local_systemone_config,
)
from arc_decision.models import ArcRecord, Choice


def record():
    return ArcRecord(
        example_id="example-1",
        dataset_id="allenai/ai2_arc",
        dataset_config="ARC-Easy",
        split="test",
        source_row_index=0,
        question="Which option is correct?",
        choices=(Choice("A", "first"), Choice("B", "second")),
        gold_answer_label="B",
        gold_answer_text="second",
    )


class FakeLayaRouter:
    def predict(self, state, questions, **kwargs):
        self.state = state
        self.questions = questions
        self.kwargs = kwargs
        return {
            "answers": {
                "arc_answer": {
                    "choice": "B",
                    "confidence": 0.8,
                    "probabilities": {"A": 0.2, "B": 0.8},
                }
            },
            "routing": {"repo": "convaiinnovations/laya"},
        }


class FakeResponse:
    headers = {}

    def __init__(self, payload):
        self.payload = payload

    def read(self):
        return json.dumps(self.payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class LocalDecisionClientTests(unittest.TestCase):
    def test_laya_runs_through_in_process_router(self):
        router = FakeLayaRouter()
        result = LayaLocalClient(router=router, checkpoint="typed-decisions").decide({"question": "test"}, record())
        self.assertEqual(result.predicted_answer_label, "B")
        self.assertEqual(result.model, "convaiinnovations/laya")
        self.assertEqual(router.questions["arc_answer"]["criteria"], {"A": "first", "B": "second"})
        self.assertEqual(router.kwargs, {"model": "typed-decisions"})

    def test_finetuned_laya_checkpoint_is_registered_and_selected(self):
        router = FakeLayaRouter()
        checkpoint_path = "/tmp/laya-arc-easy/checkpoint"
        with patch("laya.Router", return_value=router) as router_factory:
            client = LayaLocalClient(finetuned_checkpoint=checkpoint_path)

        router_factory.assert_called_once_with(
            models={"arc_finetuned": checkpoint_path},
            preload=False,
        )
        result = client.decide({"question": "test"}, record())
        self.assertEqual(router.kwargs, {"model": "arc_finetuned"})
        self.assertEqual(result.model, checkpoint_path)

    def test_local_server_defaults_need_no_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            kev = local_systemone_config("kev", timeout=5)
            clef = local_systemone_config("clef", timeout=5)
        self.assertEqual(kev.api_url, "http://127.0.0.1:8008/v1/systemone")
        self.assertEqual(kev.model_id, "jaredpalmer/kev-4b")
        self.assertIsNone(kev.api_key)
        self.assertEqual(clef.api_url, "http://127.0.0.1:30000/v1/systemone")
        self.assertEqual(clef.model_id, "Cloudflare/clef-flash")
        self.assertIsNone(clef.api_key)

        with patch.dict(os.environ, {"CLEF_MODEL_ID": "Cloudflare/clef"}, clear=True):
            full_clef = local_systemone_config("clef", timeout=5)
        self.assertEqual(full_clef.model_id, "Cloudflare/clef")

    def test_local_systemone_request_has_no_authorization_by_default(self):
        config = LocalSystemOneConfig(
            decision_model="kev",
            model_id="jaredpalmer/kev-4b",
            api_url="http://127.0.0.1:8008/v1/systemone",
            api_key=None,
            timeout=5,
        )
        response = {
            "model": "jaredpalmer/kev-4b",
            "answers": {"arc_answer": {"choice": "B", "confidence": 0.75, "probabilities": {"A": 0.25, "B": 0.75}}},
            "usage": {"input_tokens": 9, "output_tokens": 1},
        }

        def fake_urlopen(request, timeout):
            self.assertEqual(timeout, 5)
            self.assertIsNone(request.get_header("Authorization"))
            self.assertEqual(json.loads(request.data.decode("utf-8"))["model"], "jaredpalmer/kev-4b")
            return FakeResponse(response)

        with patch("arc_decision.decision_client.urlopen", side_effect=fake_urlopen):
            result = LocalSystemOneClient(config).decide({"question": "test"}, record())
        self.assertEqual(result.predicted_answer_label, "B")
        self.assertEqual(result.usage.input_tokens, 9)
