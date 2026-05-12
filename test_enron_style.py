import json
import tempfile
import unittest
import csv
import subprocess
import sys
from pathlib import Path

from enron_style import (
    build_dataset,
    build_generation_messages,
    evaluate_history,
    generate_style_response,
    infer_intent_guidance,
    infer_profile_identity,
    clean_model_response,
    extract_author_topic,
    extract_incoming_name,
    extract_reply_context,
    EmailRecord,
    enforce_outbound_recipient,
    is_author_style_prompt,
    should_use_author_style,
    looks_like_failed_generation,
    is_freeform_email_instruction,
    normalize_backend,
    retrieve_profile_examples,
    max_source_ngram_overlap,
    score_against_profile,
    has_source_copy_risk,
    style_distance_to_profile,
    select_query_records,
    select_users,
)
from lora_tools import export_lora_dataset
from lora_backend import resolve_adapter_path
from style_corpus import build_author_history, infer_passage_topic, split_passages


def write_email(path: Path, subject: str, body: str, message_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                f"Message-ID: <{message_id}>",
                "Date: Mon, 1 Jan 2001 09:00:00 -0800",
                "From: taylor.thomas@enron.com",
                "To: colleague@enron.com",
                f"Subject: {subject}",
                "",
                body,
            ]
        ),
        encoding="utf-8",
    )


class EnronStyleTests(unittest.TestCase):
    def test_selects_high_volume_t_user_first(self):
        grouped = {"tim": [object()] * 3, "tom": [object()] * 5, "amy": [object()] * 10}
        self.assertEqual(select_users(grouped, prefix="t", max_users=1, min_emails=1), ["tom"])

    def test_select_query_records_prefers_context(self):
        records = [
            EmailRecord("plain", "taylor", "Plain", "Body"),
            EmailRecord("context-1", "taylor", "Re", "Reply", "Incoming one"),
            EmailRecord("context-2", "taylor", "Re", "Reply", "Incoming two"),
        ]

        selected = select_query_records(records, 2)

        self.assertEqual([record.id for record in selected], ["context-1", "context-2"])

    def test_build_dataset_writes_expected_schemas(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for index in range(6):
                write_email(
                    root / "taylor" / "sent" / str(index),
                    subject=f"Schedule update {index}",
                    body=f"Hi Sarah,\n\nSchedule update {index} works for me.\n\nThanks,\nTaylor",
                    message_id=f"taylor-{index}",
                )
            for index in range(2):
                write_email(
                    root / "anderson" / "sent" / str(index),
                    subject=f"Ignored {index}",
                    body="This should not be selected.",
                    message_id=f"anderson-{index}",
                )

            out_dir = root / "processed"
            outputs = build_dataset(
                maildir=root,
                out_dir=out_dir,
                prefix="t",
                max_users=100,
                profile_size=4,
                query_count=2,
            )

            profile_users = json.loads(outputs["profile_user"].read_text(encoding="utf-8"))
            histories = json.loads(outputs["user_email_history"].read_text(encoding="utf-8"))
            labels = json.loads(outputs["user_email_history_label"].read_text(encoding="utf-8"))

            self.assertEqual(len(profile_users), 1)
            self.assertIn("emails are", profile_users[0]["output"])
            self.assertEqual(histories[0]["user_id"], profile_users[0]["id"])
            self.assertEqual(len(histories[0]["profile"]), 4)
            self.assertEqual(len(histories[0]["query"]), 2)
            self.assertEqual(labels["task"], "LaMP_3")
            self.assertEqual(len(labels["golds"]), 2)
            self.assertEqual(labels["golds"][0]["output"], histories[0]["query"][0]["gold"])

    def test_build_dataset_uses_reply_context_when_available(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for index in range(4):
                write_email(
                    root / "taylor" / "sent" / str(index),
                    subject=f"Profile {index}",
                    body=f"Hi Sarah,\n\nProfile email {index}.\n\nThanks,\nTaylor",
                    message_id=f"profile-{index}",
                )
            write_email(
                root / "taylor" / "sent" / "query",
                subject="Re: Files",
                body=(
                    "Please send them by Wednesday.\n\nThanks,\nTaylor\n\n"
                    "-----Original Message-----\n"
                    "From: Wendy\n"
                    "Subject: Files\n\n"
                    "When do you need these files?"
                ),
                message_id="query-context",
            )

            outputs = build_dataset(
                maildir=root,
                out_dir=root / "processed",
                prefix="t",
                max_users=1,
                profile_size=4,
                query_count=1,
            )
            histories = json.loads(outputs["user_email_history"].read_text(encoding="utf-8"))

            self.assertIn("Incoming email", histories[0]["query"][0]["input"])
            self.assertIn("When do you need these files?", histories[0]["query"][0]["input"])
            self.assertEqual(histories[0]["query"][0]["gold"], "Please send them by Wednesday.\n\nThanks,\nTaylor")

    def test_build_dataset_accepts_kaggle_emails_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            csv_path = root / "emails.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["file", "message"])
                writer.writeheader()
                for index in range(5):
                    writer.writerow(
                        {
                            "file": f"thomas-t/_sent_mail/{index}.",
                            "message": "\n".join(
                                [
                                    f"Message-ID: <csv-t-{index}>",
                                    "From: thomas.taylor@enron.com",
                                    "To: colleague@enron.com",
                                    f"Subject: CSV subject {index}",
                                    "",
                                    "Hi Pat,\n\nCSV sent mail works.\n\nThanks,\nTaylor",
                                ]
                            ),
                        }
                    )
                writer.writerow(
                    {
                        "file": "thomas-t/inbox/ignored.",
                        "message": "Subject: ignored\n\nThis inbox message should not be used.",
                    }
                )

            outputs = build_dataset(
                maildir=root,
                out_dir=root / "processed",
                prefix="t",
                max_users=1,
                profile_size=4,
                query_count=1,
            )
            histories = json.loads(outputs["user_email_history"].read_text(encoding="utf-8"))

            self.assertEqual(len(histories), 1)
            self.assertIn("source_user", histories[0])
            self.assertIn("inferred_name", histories[0])
            self.assertEqual(len(histories[0]["profile"]), 4)
            self.assertEqual(histories[0]["query"][0]["gold"], "Hi Pat,\n\nCSV sent mail works.\n\nThanks,\nTaylor")

    def test_generation_uses_profile_style_without_ollama(self):
        profile = [
            {
                "id": "e001",
                "subject": "Schedule",
                "body": "Hi Sarah,\n\nThat time works for me.\n\nThanks,\nTaylor",
            },
            {
                "id": "e002",
                "subject": "Review",
                "body": "Hi Mark,\n\nI will review this today.\n\nThanks,\nTaylor",
            },
        ]

        response = generate_style_response(
            profile=profile,
            prompt="Write a reply to this email in the user's style: Can we reschedule Friday's call?",
            use_ollama=False,
        )

        self.assertNotIn("Sarah", response)
        self.assertIn("works for me", response)
        self.assertIn("Thanks", response)

    def test_generation_does_not_reuse_long_first_sentence_as_greeting(self):
        profile = [
            {
                "id": "e001",
                "subject": "Old mail",
                "body": (
                    "Hi George.  I'm sorry I didn't get back to you sooner.  "
                    "I was just going through some old e-mails and found this one."
                ),
            }
        ]

        response = generate_style_response(
            profile=profile,
            prompt="How are you doing?",
            use_ollama=False,
        )

        self.assertNotIn("old e-mails", response)
        self.assertLessEqual(len(response.splitlines()[0]), 40)

    def test_generation_answers_social_prompt_without_random_profile_name(self):
        profile = [
            {
                "id": "e001",
                "subject": "Old mail",
                "body": "David,\n\nThanks for sending this over.",
            }
        ]

        response = generate_style_response(
            profile=profile,
            prompt="How are you doing?",
            use_ollama=False,
        )

        self.assertNotIn("David", response)
        self.assertIn("Doing well", response)

    def test_generation_uses_prompt_recipient_when_present(self):
        profile = [
            {
                "id": "e001",
                "subject": "Old mail",
                "body": "David,\n\nThanks for sending this over.",
            }
        ]

        response = generate_style_response(
            profile=profile,
            prompt="Write to Sarah: How are you doing?",
            use_ollama=False,
        )

        self.assertTrue(response.startswith("Sarah,"))

    def test_generation_handles_deadline_file_prompt_without_ollama(self):
        profile = [
            {
                "id": "e001",
                "subject": "Files",
                "body": "Wendy,\n\nPlease send the files over when you can.\n\nThanks,\nTaylor",
            }
        ]

        response = generate_style_response(
            profile=profile,
            prompt="Write me an email to Wendy telling her we need to get these files in by Wednesday.",
            use_ollama=False,
        )

        self.assertIn("Wendy", response)
        self.assertIn("files", response.lower())
        self.assertIn("Wednesday", response)

    def test_author_style_fallback_generates_recognizable_passage(self):
        profile = [
            {
                "id": "p001",
                "subject": "Education",
                "body": (
                    "Upon the subject of education, not presuming to dictate any plan, "
                    "I can only say that I view it as the most important subject which "
                    "we, as a people, can be engaged in."
                ),
            },
            {
                "id": "p002",
                "subject": "Free institutions",
                "body": (
                    "Let reverence for the laws be breathed by every American mother "
                    "to the lisping babe that prattles on her lap."
                ),
            },
        ]

        response = generate_style_response(
            profile=profile,
            prompt="Write a new passage in Abraham Lincoln's style inspired by this topic: education matters. Do not quote or continue an existing source passage.",
            use_ollama=False,
        )
        scores = score_against_profile(response, profile)

        self.assertIn("Upon the subject of why education matters", response)
        self.assertIn("free", response.lower())
        self.assertGreater(len(response.split()), 70)
        self.assertLessEqual(scores["profile_copy_5gram_rate"], 0.15)

    def test_extract_author_topic_from_plain_prompt(self):
        self.assertEqual(
            extract_author_topic("Write a brief public statement about why education matters."),
            "why education matters",
        )
        self.assertEqual(
            extract_author_topic("Write a short note encouraging a team to finish their reports by Friday."),
            "finishing the reports by Friday",
        )

    def test_author_style_router_accepts_freeform_voice_prompt(self):
        self.assertTrue(
            should_use_author_style(
                "Write a short note encouraging a team to finish reports by Friday.",
                "Abraham Lincoln",
            )
        )
        self.assertFalse(
            should_use_author_style(
                "Write me an email to Wendy telling her the files are due Friday.",
                "Taylor",
            )
        )

    def test_generation_answers_name_from_profile_identity_without_ollama(self):
        profile = [
            {
                "id": "e001",
                "subject": "Update",
                "body": "Please review when you can.\n\nThanks,\nTaylor",
            },
            {
                "id": "e002",
                "subject": "Schedule",
                "body": "That works for me.\n\nTaylor",
            },
        ]

        response = generate_style_response(
            profile=profile,
            prompt="What's your name?",
            use_ollama=False,
        )

        self.assertEqual(response, "My name is Taylor.")

    def test_generation_answers_introduction_prompt_naturally_without_ollama(self):
        profile = [
            {
                "id": "e001",
                "subject": "Update",
                "body": "Please review when you can.\n\nThanks,\nTaylor",
            }
        ]

        response = generate_style_response(
            profile=profile,
            prompt="Hi, my name is Sarah. What's your name?",
            use_ollama=False,
        )

        self.assertEqual(response, "Hi Sarah, my name is Taylor.")

    def test_generation_prompt_tells_llama_not_to_copy_example_facts(self):
        messages = build_generation_messages(
            prompt="What's your name?",
            style_hint="The user's emails are brief.",
            fingerprint={"average_words": 6, "short_reply_examples": ["OK by me."]},
            identity="Taylor",
            examples=[
                {
                    "id": "e001",
                    "subject": "Old mail",
                    "body": "Hi George. Sorry I did not get back to you sooner.",
                }
            ],
        )
        combined = "\n".join(message["content"] for message in messages)

        self.assertIn("Do not copy facts", combined)
        self.assertIn("text after the colon as", combined)
        self.assertIn("Do not claim you have completed", combined)
        self.assertIn("Never include a preamble", combined)
        self.assertIn("Intent guidance", combined)
        self.assertIn("Inferred user identity: Taylor", combined)
        self.assertIn("What's your name?", combined)

    def test_author_style_prompt_uses_generic_requested_text_language(self):
        prompt = "Write a new passage in Lincoln's style inspired by this topic: patience."
        messages = build_generation_messages(
            prompt=prompt,
            style_hint="Formal and reflective.",
            fingerprint={"average_words": 60},
            identity="Lincoln",
            examples=[{"id": "p001", "subject": "Patience", "body": "Let us be patient and firm."}],
        )
        combined = "\n".join(message["content"] for message in messages)

        self.assertTrue(is_author_style_prompt(prompt))
        self.assertIn("You write original public prose", combined)
        self.assertIn("Write only the passage", combined)

    def test_clean_model_response_removes_preamble_and_subject(self):
        cleaned = clean_model_response(
            "It looks like I'm supposed to write a reply as Mark. Here it is:\n\n"
            "Subject: RE: Itinerary\n\nI'll check with travel."
        )

        self.assertEqual(cleaned, "I'll check with travel.")

    def test_infer_intent_guidance_for_review_request(self):
        guidance = infer_intent_guidance(
            "Write a reply to Alex: Please review the attached draft agreement."
        )

        self.assertIn("will review", guidance)
        self.assertIn("do not say you already reviewed", guidance)

    def test_infer_intent_guidance_for_introduction(self):
        guidance = infer_intent_guidance("Hi, my name is Sarah. What's your name?", "Taylor")

        self.assertIn("introducing themselves", guidance)
        self.assertIn("Taylor", guidance)

    def test_extract_incoming_name(self):
        self.assertEqual(extract_incoming_name("Hi, my names Sarah, what's your name?"), "Sarah")
        self.assertEqual(extract_incoming_name("Hi, my names mark, what your name?"), "Mark")

    def test_extract_reply_context_from_original_message(self):
        context = extract_reply_context(
            "My reply.\n\n-----Original Message-----\nFrom: Wendy\n\nWhen do you need these files?"
        )

        self.assertIn("From: Wendy", context)
        self.assertIn("When do you need these files?", context)

    def test_name_question_accepts_informal_wording(self):
        profile = [
            {
                "id": "e001",
                "subject": "Update",
                "body": "Please review when you can.\n\nThanks,\nTaylor",
            }
        ]

        response = generate_style_response(
            profile=profile,
            prompt="Hi, my names mark, what your name?",
            use_ollama=False,
        )

        self.assertEqual(response, "Hi Mark, my name is Taylor.")

    def test_infer_profile_identity_ignores_original_message_artifact(self):
        profile = [
            {"id": "e001", "subject": "", "body": "Looks good.\n\nOriginal Message"},
            {"id": "e002", "subject": "", "body": "I agree.\n\nMark Taylor"},
        ]

        self.assertEqual(infer_profile_identity(profile), "Mark Taylor")

    def test_infer_profile_identity_uses_common_signature(self):
        profile = [
            {"id": "e001", "subject": "", "body": "Looks good.\n\nThanks,\nTaylor"},
            {"id": "e002", "subject": "", "body": "I agree.\n\nTaylor"},
            {"id": "e003", "subject": "", "body": "Please call Mark to discuss."},
        ]

        self.assertEqual(infer_profile_identity(profile), "Taylor")

    def test_evaluate_history_returns_metrics(self):
        histories = [
            {
                "user_id": 20000001,
                "profile": [
                    {
                        "id": "e001",
                        "subject": "Schedule",
                        "body": "Hi Sarah,\n\nThat time works for me.\n\nThanks,\nTaylor",
                    }
                ],
                "query": [
                    {
                        "id": "q001",
                        "input": "Write a reply in the user's style: Can we reschedule?",
                        "gold": "Hi Sarah,\n\nThat works for me.\n\nThanks,\nTaylor",
                    }
                ],
            }
        ]

        report = evaluate_history(histories)

        self.assertEqual(report["count"], 1)
        self.assertIn("word_f1", report["metrics"])
        self.assertIn("profile_copy_5gram_rate", report["metrics"])
        self.assertIn("profile_longest_copy_run", report["metrics"])
        self.assertIn("style_distance", report["metrics"])
        self.assertEqual(len(report["examples"]), 1)

    def test_profile_copy_metric_flags_direct_source_copying(self):
        sources = [
            "This is a uniquely identifying passage about the quarterly storage model and hedging detail."
        ]

        copied = max_source_ngram_overlap(
            "Please note the uniquely identifying passage about the quarterly storage model today.",
            sources,
            n=5,
        )
        paraphrased = max_source_ngram_overlap(
            "Please note the storage model needs another quarterly review today.",
            sources,
            n=5,
        )

        self.assertGreater(copied, 0.2)
        self.assertEqual(paraphrased, 0.0)

    def test_source_copy_risk_flags_long_source_overlap(self):
        profile = [
            {
                "id": "p001",
                "subject": "Education",
                "body": "Upon the subject of education not presuming to dictate any plan or system respecting it I can only say this matters.",
            }
        ]

        self.assertTrue(
            has_source_copy_risk(
                "Upon the subject of education not presuming to dictate any plan or system respecting it.",
                profile,
            )
        )
        self.assertFalse(
            has_source_copy_risk(
                "Education gives citizens the power to judge public questions with steadier minds.",
                profile,
            )
        )

    def test_generation_avoids_copying_unique_profile_language_for_new_prompt(self):
        profile = [
            {
                "id": "e001",
                "subject": "Storage model",
                "body": (
                    "Alex,\n\nThe blue falcon storage hedge memo contains the cactus lantern clause "
                    "and should stay with legal.\n\nThanks,\nTaylor"
                ),
            }
        ]

        response = generate_style_response(
            profile=profile,
            prompt="Write me an email to Wendy telling her we need the files by Wednesday.",
            use_ollama=False,
        )
        profile_scores = score_against_profile(response, profile)

        self.assertNotIn("cactus lantern", response.lower())
        self.assertLessEqual(profile_scores["profile_copy_5gram_rate"], 0.15)
        self.assertLessEqual(profile_scores["profile_longest_copy_run"], 4)

    def test_style_distance_rewards_matching_length_and_punctuation(self):
        profile = [
            {"id": "e001", "subject": "", "body": "Fine by me.\n\nTaylor"},
            {"id": "e002", "subject": "", "body": "Looks good.\n\nTaylor"},
        ]

        near = style_distance_to_profile("Works for me.\n\nTaylor", profile)
        far = style_distance_to_profile(
            "I have reviewed the entire package, and after considering several possible alternatives, "
            "I think we should arrange a long meeting to discuss every open issue in detail.",
            profile,
        )

        self.assertLess(near, far)

    def test_normalize_backend_accepts_aliases(self):
        self.assertEqual(normalize_backend("hf", use_ollama=False), "peft")
        self.assertEqual(normalize_backend("llama", use_ollama=False), "ollama")
        self.assertEqual(normalize_backend("", use_ollama=False), "fallback")
        self.assertEqual(normalize_backend("", use_ollama=True), "ollama")

    def test_enhanced_retrieval_prefers_subject_and_intent_match(self):
        profile = [
            {
                "id": "later",
                "subject": "Travel plans",
                "body": "I will book the flight and hotel today.",
            },
            {
                "id": "deadline",
                "subject": "Files due Wednesday",
                "body": "Please send the files by Wednesday so we can review them.",
            },
        ]

        retrieved = retrieve_profile_examples(
            profile,
            "Write to Wendy saying we need the files by Wednesday.",
            top_k=1,
        )

        self.assertEqual(retrieved[0]["id"], "deadline")

    def test_export_lora_dataset_keeps_validation_on_query_rows_by_default(self):
        histories = [
            {
                "user_id": 20000001,
                "source_user": "taylor",
                "inferred_name": "Taylor",
                "profile": [
                    {
                        "id": "p001",
                        "subject": "Schedule",
                        "body": "Hi Sarah,\n\nThat time works for me.\n\nThanks,\nTaylor",
                    },
                    {
                        "id": "p002",
                        "subject": "Review",
                        "body": "Mark,\n\nI will review this today.\n\nTaylor",
                    },
                ],
                "query": [
                    {
                        "id": "q001",
                        "input": "Write a reply in the user's style: Can we reschedule?",
                        "gold": "That works for me.\n\nTaylor",
                    },
                    {
                        "id": "q002",
                        "input": "Write a reply in the user's style: Please review this.",
                        "gold": "I will review this today.\n\nTaylor",
                    },
                ],
            }
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            history_path = root / "history.json"
            history_path.write_text(json.dumps(histories), encoding="utf-8")
            manifest = export_lora_dataset(
                history_path=history_path,
                out_dir=root / "lora",
                val_ratio=0.5,
                include_profile=True,
            )
            val_rows = [
                json.loads(line)
                for line in Path(manifest["val_path"]).read_text(encoding="utf-8").splitlines()
            ]

        self.assertGreater(manifest["profile_rows"], 0)
        self.assertEqual({row["source"] for row in val_rows}, {"query"})

    def test_export_lora_dataset_can_write_per_user_splits(self):
        histories = [
            {
                "user_id": 20000001,
                "source_user": "taylor",
                "inferred_name": "Taylor",
                "profile": [
                    {
                        "id": "p001",
                        "subject": "Schedule",
                        "body": "Hi Sarah,\n\nThat time works for me.\n\nThanks,\nTaylor",
                    }
                ],
                "query": [
                    {
                        "id": "q001",
                        "input": "Write a reply in the user's style: Can we reschedule?",
                        "gold": "That works for me.\n\nTaylor",
                    }
                ],
            }
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            history_path = root / "history.json"
            history_path.write_text(json.dumps(histories), encoding="utf-8")
            manifest = export_lora_dataset(
                history_path=history_path,
                out_dir=root / "lora",
                val_ratio=0.5,
                include_profile=True,
                per_user=True,
            )
            user_manifest = manifest["users"]["20000001"]

            self.assertTrue(Path(user_manifest["train_path"]).exists())
            self.assertTrue(Path(user_manifest["val_path"]).exists())

    def test_resolve_adapter_path_prefers_explicit_path_then_user_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            explicit = root / "explicit"
            user_dir = root / "adapters" / "20000001"
            explicit.mkdir()
            user_dir.mkdir(parents=True)

            self.assertEqual(resolve_adapter_path(str(explicit), str(root / "adapters"), 20000001), str(explicit))
            self.assertEqual(resolve_adapter_path("", str(root / "adapters"), 20000001), str(user_dir))
            self.assertEqual(resolve_adapter_path("", str(root / "missing"), 20000001), "")

    def test_per_person_adapter_paths_are_separate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "adapters" / "lincoln"
            second = root / "adapters" / "shakespeare"
            first.mkdir(parents=True)
            second.mkdir(parents=True)

            self.assertEqual(resolve_adapter_path("", str(root / "adapters"), "lincoln"), str(first))
            self.assertEqual(resolve_adapter_path("", str(root / "adapters"), "shakespeare"), str(second))
            self.assertNotEqual(first, second)

    def test_project_generate_cli_runs_fallback(self):
        histories = [
            {
                "user_id": 20000001,
                "source_user": "taylor",
                "inferred_name": "Taylor",
                "profile": [
                    {
                        "id": "p001",
                        "subject": "Files",
                        "body": "Wendy,\n\nPlease send the files over when you can.\n\nThanks,\nTaylor",
                    }
                ],
                "query": [],
            }
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            history_path = Path(temp_dir) / "history.json"
            history_path.write_text(json.dumps(histories), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    "project.py",
                    "generate",
                    "--history",
                    str(history_path),
                    "--backend",
                    "fallback",
                    "--no-ollama",
                    "--prompt",
                    "Write me an email to Wendy telling her we need the files by Wednesday.",
                ],
                cwd=Path(__file__).parent,
                text=True,
                capture_output=True,
                check=True,
            )

        self.assertIn("Wendy", result.stdout)
        self.assertIn("Wednesday", result.stdout)

    def test_failed_generation_detector_catches_bad_lora_output(self):
        self.assertTrue(looks_like_failed_generation("W论文!"))
        self.assertTrue(looks_like_failed_generation("Equal justice!"))
        self.assertTrue(looks_like_failed_generation("factors factors factors factors factors factors factors factors factors factors factors factors"))
        self.assertFalse(looks_like_failed_generation("Wendy,\n\nWe need to get those files in by Wednesday."))

    def test_freeform_email_instruction_detection(self):
        self.assertTrue(
            is_freeform_email_instruction(
                "Write me an email to Wendy telling her we need these files by Wednesday."
            )
        )
        self.assertFalse(
            is_freeform_email_instruction(
                "Write a reply to this email in the user's style.\n\nSubject: RE: Files\n\nIncoming email:"
            )
        )

    def test_enforce_outbound_recipient_prepends_missing_greeting(self):
        text = enforce_outbound_recipient(
            "We need to get those files in by Wednesday.",
            "Write me an email to Wendy telling her we need these files by Wednesday.",
        )

        self.assertTrue(text.startswith("Wendy,\n\n"))

    def test_split_passages_creates_trainable_chunks(self):
        text = (
            "First passage has enough words to be useful for training and evaluation. "
            "It continues with measured language and a second sentence.\n\n"
            "Second passage also has enough words to become a separate style example. "
            "It gives the splitter something clear to preserve."
        )

        passages = split_passages(text, min_words=12, max_words=40)

        self.assertEqual(len(passages), 2)
        self.assertTrue(all(len(p.split()) >= 12 for p in passages))

    def test_split_passages_strips_gutenberg_boilerplate(self):
        text = (
            "Header license words should disappear.\n"
            "*** START OF THE PROJECT GUTENBERG EBOOK SAMPLE ***\n\n"
            "Editor introduction should disappear.\n\n"
            "LINCOLN'S SPEECHES AND LETTERS\n\n"
            "This real passage contains enough words to become a useful training example "
            "after the boilerplate has been removed from the corpus.\n\n"
            "*** END OF THE PROJECT GUTENBERG EBOOK SAMPLE ***\n"
            "Footer license words should disappear."
        )

        passages = split_passages(text, min_words=10, max_words=80)

        self.assertEqual(len(passages), 1)
        self.assertIn("real passage", passages[0])
        self.assertNotIn("Header license", passages[0])
        self.assertNotIn("Editor introduction", passages[0])
        self.assertNotIn("Footer license", passages[0])

    def test_build_author_history_uses_shared_profile_query_schema(self):
        document_text = "\n\n".join(
            f"Passage {index} contains enough thoughtful language to become a useful training example. "
            "It has a steady rhythm and avoids being too short for the splitter."
            for index in range(1, 7)
        )

        history = build_author_history(
            documents=[{"title": "Collected Notes", "text": document_text}],
            author_id="lincoln",
            display_name="Abraham Lincoln",
            profile_size=4,
            query_count=2,
            min_words=12,
            max_words=60,
        )

        self.assertEqual(history["user_id"], "lincoln")
        self.assertEqual(history["inferred_name"], "Abraham Lincoln")
        self.assertEqual(len(history["profile"]), 4)
        self.assertEqual(len(history["query"]), 2)
        self.assertIn("Do not quote", history["query"][0]["input"])

    def test_author_history_extracts_passage_heading_as_subject(self):
        text = (
            "_Short Address. March 9, 1832_\n\n"
            "This passage contains enough thoughtful language to become a useful training example. "
            "It has a steady rhythm and avoids being too short for the splitter.\n\n"
            "_Another Address. April 1, 1832_\n\n"
            "This second passage contains enough thoughtful language to become a useful training example. "
            "It has a steady rhythm and avoids being too short for the splitter."
        )

        history = build_author_history(
            documents=[{"title": "Fallback Title", "text": text}],
            author_id="lincoln",
            display_name="Abraham Lincoln",
            profile_size=1,
            query_count=1,
            min_words=12,
            max_words=60,
        )

        self.assertEqual(history["profile"][0]["subject"], "Short Address. March 9, 1832")
        self.assertNotIn("Short Address", history["profile"][0]["body"])
        self.assertIn("Another Address", history["query"][0]["input"])

    def test_infer_passage_topic_uses_opening_words(self):
        topic = infer_passage_topic(
            "upon the subject of education, not presuming to dictate any plan or system respecting it"
        )

        self.assertEqual(topic, "Upon the subject of education not presuming to dictate any plan or")


if __name__ == "__main__":
    unittest.main()
