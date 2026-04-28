import json
import tempfile
import unittest
import csv
from pathlib import Path

from enron_style import (
    build_dataset,
    build_generation_messages,
    evaluate_history,
    generate_style_response,
    infer_intent_guidance,
    infer_profile_identity,
    clean_model_response,
    extract_incoming_name,
    extract_reply_context,
    EmailRecord,
    select_query_records,
    select_users,
)


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
        self.assertEqual(len(report["examples"]), 1)


if __name__ == "__main__":
    unittest.main()
