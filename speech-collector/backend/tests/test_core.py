import sqlite3
import unittest
from pathlib import Path

from app.audio import convert_to_training_wav
from app.dictionary_importer import DictionarySourceSpec, entries_from_page
from app.db import SCHEMA
from app.invitations import generate_invitation_code, invitation_is_usable
from app.manifest import manifest_rows, render_csv, render_jsonl
from app.task_importer import tasks_from_rime, tasks_from_tsv


class CoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.work = Path("tests_work")
        self.work.mkdir(exist_ok=True)

    def test_tasks_from_rime_reads_body_after_marker(self) -> None:
        source = self.work / "sample.dict.yaml"
        source.write_text(
            """# comment
---
name: sample
...
一\tia3\t1
今日天色冷
""",
            encoding="utf-8",
        )

        rows = tasks_from_rime(source, "ruian", "word", "sample", None)

        self.assertEqual(rows[0]["text"], "一")
        self.assertEqual(rows[0]["romanization"], "ia3")
        self.assertEqual(rows[1]["type"], "word")
        self.assertEqual(rows[1]["text"], "今日天色冷")

    def test_tasks_from_tsv_uses_explicit_fields(self) -> None:
        source = self.work / "sentences.tsv"
        source.write_text(
            "text\tromanization\ttype\tsource\tpriority\tstatus\n請慢慢講清楚\t\tsentence\tmanual\t9\tready\n",
            encoding="utf-8",
        )

        rows = tasks_from_tsv(source, "wenzhou")

        self.assertEqual(rows[0]["dialect"], "wenzhou")
        self.assertEqual(rows[0]["type"], "sentence")
        self.assertEqual(rows[0]["text"], "請慢慢講清楚")
        self.assertEqual(rows[0]["priority"], 9)

    def test_manifest_exports_only_approved_by_default(self) -> None:
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.executescript(SCHEMA)
        conn.execute("INSERT INTO invitations(code) VALUES ('TEST')")
        conn.execute(
            """
            INSERT INTO speakers(id, invite_code, region, age_group, dialect_point, consent_version)
            VALUES ('spk', 'TEST', '瑞安', '30-39', '瑞安', 'research-consent-v1')
            """
        )
        conn.execute(
            """
            INSERT INTO tasks(id, dialect, type, text, romanization, source)
            VALUES ('task', 'ruian', 'word', '一', 'ia3', 'unit')
            """
        )
        conn.execute(
            """
            INSERT INTO submissions(
                id, invite_code, speaker_id, task_id, dialect, raw_audio_path,
                wav_audio_path, duration_seconds, consent_version, review_status
            )
            VALUES ('sub', 'TEST', 'spk', 'task', 'ruian', 'raw.webm', 'one.wav', 1.25,
                    'research-consent-v1', 'approved')
            """
        )

        rows = manifest_rows(conn)

        self.assertEqual(rows[0]["audio_filepath"], "one.wav")
        self.assertEqual(rows[0]["text"], "一")
        self.assertIn('"dialect": "ruian"', render_jsonl(rows))
        self.assertIn("audio_filepath,text,dialect", render_csv(rows))

    def test_convert_to_training_wav_returns_false_for_missing_input(self) -> None:
        result = convert_to_training_wav(self.work / "missing.webm", self.work / "out.wav")

        self.assertFalse(result)

    def test_invitation_code_generation_and_usage_limit(self) -> None:
        code = generate_invitation_code("ruian")
        self.assertTrue(code.startswith("RA-"))
        row = {"active": 1, "max_uses": 1, "used_count": 1, "expires_at": ""}
        self.assertFalse(invitation_is_usable(row))

    def test_dictionary_entries_keep_source_page_and_pending_status(self) -> None:
        source = DictionarySourceSpec("温州方言读音字典", "", Path("sample.pdf"), "wenzhou")
        rows = entries_from_page(source, 12, "東 dong 釋義一\n今日天色冷 例句")

        self.assertEqual(rows[0]["source"], "温州方言读音字典")
        self.assertEqual(rows[0]["page"], 12)
        self.assertEqual(rows[0]["review_status"], "pending")
        self.assertEqual(rows[0]["dialect"], "wenzhou")


if __name__ == "__main__":
    unittest.main()
