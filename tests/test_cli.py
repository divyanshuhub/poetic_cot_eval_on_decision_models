import unittest
from pathlib import Path
from unittest.mock import patch

from arc_decision.cli import _config_from_args, _project_root, _resolve_output_dir, build_parser


class OutputDirectoryTests(unittest.TestCase):
    def test_relative_path_resolves_from_project_root(self):
        project_root = _project_root()
        self.assertEqual(
            _resolve_output_dir("outputs/run-a"),
            project_root / "outputs" / "run-a",
        )

    def test_absolute_path_inside_project_is_allowed(self):
        project_root = _project_root()
        output_dir = project_root / "outputs" / "run-a"
        self.assertEqual(_resolve_output_dir(output_dir), output_dir)

    def test_path_outside_project_is_rejected(self):
        outside_path = _project_root().parent / "outside-run"
        with self.assertRaisesRegex(ValueError, "inside the project folder"):
            _resolve_output_dir(outside_path)

    def test_project_root_itself_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be a subdirectory"):
            _resolve_output_dir(_project_root())


class ConfigDefaultTests(unittest.TestCase):
    def test_yaml_supplies_output_dir_and_gemini_key_variable(self):
        args = build_parser().parse_args(["generate"])
        with patch(
            "arc_decision.cli._load_defaults",
            return_value={"output_dir": "outputs/from-yaml", "gemini_api_key_env": "GEMINI_API_KEY_2"},
        ):
            config = _config_from_args(args)

        self.assertEqual(config.output_dir, _project_root() / "outputs" / "from-yaml")
        self.assertEqual(config.gemini_api_key_env, "GEMINI_API_KEY_2")

    def test_cli_output_dir_overrides_yaml_default(self):
        args = build_parser().parse_args(["generate", "--output-dir", "outputs/from-cli"])
        with patch(
            "arc_decision.cli._load_defaults",
            return_value={"output_dir": "outputs/from-yaml"},
        ):
            config = _config_from_args(args)

        self.assertEqual(config.output_dir, _project_root() / "outputs" / "from-cli")

    def test_relative_laya_finetuned_checkpoint_resolves_from_project_root(self):
        args = build_parser().parse_args(
            [
                "evaluate",
                "--laya-finetuned-checkpoint",
                "outputs/finetunes/laya_arc_easy/checkpoint",
            ]
        )
        with patch("arc_decision.cli._load_defaults", return_value={}):
            config = _config_from_args(args)

        self.assertEqual(
            config.laya_finetuned_checkpoint,
            str(_project_root() / "outputs" / "finetunes" / "laya_arc_easy" / "checkpoint"),
        )

    def test_laya_finetuned_checkpoint_rejects_other_decision_models(self):
        args = build_parser().parse_args(
            ["evaluate", "--decision-model", "kev", "--laya-finetuned-checkpoint", "checkpoint"]
        )
        with patch("arc_decision.cli._load_defaults", return_value={}):
            with self.assertRaisesRegex(ValueError, "only be used with decision_model='laya'"):
                _config_from_args(args)


if __name__ == "__main__":
    unittest.main()