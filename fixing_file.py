### === PART 1/3 START ===
import io
import logging
import base64
import json
import os
import string
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from openai import OpenAI
from pydub import AudioSegment
from app.speech.process import perform_asr_with_word_timestamps, get_tts
from app.speech.utils import PauseStats
import difflib
from jiwer import wer
import numpy as np

from app.toefl_report.types import (
    DiscourseEvaluation,
    GrammarError,
    GrammarFormatDiff,
    InterviewFluency,
    InterviewGrammar,
    InterviewResult,
    ListenRepeatFluency,
    ListenRepeatGrammar,
    ListenRepeatPerPairFluency,
    ListenRepeatResult,
    Mispronunciations,
    PairError,
    PerPairFluency,
    ToeflPairEvaluationResult,
    ToeflPairType,
    ToeflReport,
    ToeflScore,
    Vocabulary,
    VocabularyAndRelevance,
    Pronunciation,
)
from app.utils.downloader import save_audio
from app.utils.s3 import uploadObjectToS3
from app.utils.alicloud import upload_audio_bytes_to_oss_direct
from redis_client import RedisClient, standard_pron_redis_key
from app.toefl_report.summaries import (
    get_fluency_summary,
    get_pronunciation_summary,
    get_grammar_summary,
)
from exceptions import GrammarAPIError

logger = logging.getLogger(__name__)

# ==========================================================
# ✅ [CHANGE: Added inline GPT-4o grammar scoring function]
# ==========================================================

def _gpt4o_grammar_feedback(answer_text: str) -> List[GrammarError]:
    """
    Calls GPT-4o to analyze grammar mistakes and return a structured GrammarError list.
    Mirrors what get_grammar_feedback_api() used to do, but runs inline.
    """
    if not answer_text.strip():
        return []

    client = OpenAI()

    system_msg = {
        "role": "system",
        "content": (
            "You are a precise English grammar evaluator. "
            "Given a student's spoken response transcript, identify grammatical errors "
            "and suggest corrections. Return valid JSON: "
            "{'feedback':[{'original_sentence':str,'corrected_sentence':str,"
            "'error_type':str,'error_type_summary':str,'error_cefr_level':str}]}"
        ),
    }

    user_msg = {
        "role": "user",
        "content": f"Evaluate the following text for grammar errors:\n{answer_text}",
    }

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[system_msg, user_msg],
            temperature=0,
        )
        content = response.choices[0].message.content.strip()
        try:
            parsed = json.loads(content)
            feedback = parsed.get("feedback", [])
        except Exception:
            # fallback parse if text formatting slightly off
            cleaned = content.strip("`").replace("json", "").strip()
            feedback = json.loads(cleaned).get("feedback", [])

        grammar_errors = []
        for g in feedback:
            fdiff_obj = GrammarFormatDiff(
                corr=g.get("corrected_sentence", ""),
                error_type=g.get("error_type", ""),
                error_type_summary=g.get("error_type_summary", ""),
                error_type_summary_cn="",
                error_cefr_level=g.get("error_cefr_level", ""),
                feedback=g.get("error_type_summary", ""),
                feedback_cn="",
                has_error=True,
                orig=g.get("original_sentence", ""),
            )
            grammar_errors.append(
                GrammarError(
                    corrected_sentence=g.get("corrected_sentence", ""),
                    fdiff=[fdiff_obj],
                    original_sentence=g.get("original_sentence", ""),
                    has_error=True,
                )
            )
        return grammar_errors

    except Exception as e:
        logger.error(f"GPT-4o grammar feedback failed: {e}")
        return []

# ==========================================================
# TOEFL Pair Evaluation
# ==========================================================

def evaluate_toefl_pair(
    prompt_audio_link: str, answer_audio_link: str, pair_type: ToeflPairType, idx: int
) -> ToeflPairEvaluationResult:
    """Evaluate a TOEFL pair and return a ToeflPairEvaluationResult."""
    try:
        prompt_audio_file_path, _, _ = save_audio(
            prompt_audio_link, use_downloader_api_for_cn=True
        )
        answer_audio_file_path, _, _ = save_audio(
            answer_audio_link, use_downloader_api_for_cn=True
        )

        prompt_audio: AudioSegment = AudioSegment.from_file(prompt_audio_file_path)
        answer_audio: AudioSegment = AudioSegment.from_file(answer_audio_file_path)
        prompt_audio_duration: float = prompt_audio.duration_seconds
        answer_audio_duration: float = answer_audio.duration_seconds

        prompt_audio_buffer = io.BytesIO()
        answer_audio_buffer = io.BytesIO()
        prompt_audio.export(prompt_audio_buffer, format="wav")
        answer_audio.export(answer_audio_buffer, format="wav")

        prompt_asr_result = perform_asr_with_word_timestamps(prompt_audio_buffer)
        answer_asr_result = perform_asr_with_word_timestamps(answer_audio_buffer)

        pause_stats = PauseStats.from_words(
            answer_asr_result.words, answer_audio_duration
        )

        repeat_accuracy, incorrect_segments = _repeat_accuracy_and_incorrect_segments(
            prompt_asr_result.text, answer_asr_result.text
        )

        # mispronunciations
        mispron_words = _gpt4o_mispronunciations(
            prompt_audio_buffer,
            answer_audio_buffer,
            prompt_asr_result.text,
            answer_asr_result.text,
        )
        mispronunciations = _segment_and_upload_mispronunciations(
            pair_id=idx,
            mispronounced_words=mispron_words,
            student_audio_buffer=answer_audio_buffer,
            use_cn_uploader=True,
        )

        grammar_errors = []
        if pair_type == ToeflPairType.INTERVIEW:
            logger.info("grammar feedback started")
            try:
                # ✅ [CHANGE: now uses inline GPT-4o scoring instead of fallback]
                grammar_errors = _gpt4o_grammar_feedback(answer_asr_result.text)
            except Exception as e:
                logger.exception(e)
                raise GrammarAPIError

        return ToeflPairEvaluationResult(
            is_error=False,
            prompt_audio_duration=prompt_audio_duration,
            answer_audio_duration=answer_audio_duration,
            repeat_accuracy=repeat_accuracy,
            incorrect_segments=incorrect_segments,
            mispronunciations=mispronunciations,
            pause_stats=pause_stats,
            prompt_audio_asr_text=prompt_asr_result.text,
            answer_audio_asr_text=answer_asr_result.text,
            pair_type=pair_type,
            idx=idx,
            prompt_audio_link=prompt_audio_link,
            answer_audio_link=answer_audio_link,
            grammar_feedback=grammar_errors,
        )
    except Exception as e:
        logger.error(f"Error evaluating TOEFL pair: {e}")
        return ToeflPairEvaluationResult(
            is_error=True,
            error_message=str(e),
            prompt_audio_duration=0,
            prompt_audio_asr_text="",
            answer_audio_duration=0,
            answer_audio_asr_text="",
            pair_type=pair_type,
            idx=idx,
            prompt_audio_link=prompt_audio_link,
            answer_audio_link=answer_audio_link,
            repeat_accuracy=0,
            incorrect_segments=[],
            mispronunciations=[],
            pause_stats=PauseStats(),
        )
    finally:
        if os.path.exists(prompt_audio_file_path):
            os.remove(prompt_audio_file_path)
        if os.path.exists(answer_audio_file_path):
            os.remove(answer_audio_file_path)
### === PART 1/3 END ===


### === PART 2/3 START ===

# ==========================================================
# TOEFL Score Computation
# ==========================================================

def _calculate_overall_score(speech_rate: int, accuracy: int) -> ToeflScore:
    """Calculate overall TOEFL score based on speech rate and accuracy."""
    speech_level = _level_from_speech_rate(speech_rate)
    accuracy_level = _level_from_accuracy(accuracy)

    levels = ["A1", "A2", "B1", "B2", "C1", "C2"]
    speech_idx = levels.index(speech_level)
    accuracy_idx = levels.index(accuracy_level)
    cefr = levels[min(speech_idx, accuracy_idx)]

    toefl_score = _cefr_to_toefl(cefr)
    old_toefl_score = int(round(accuracy * 0.3))

    return ToeflScore(
        cefr=cefr, toefl_score=toefl_score, old_toefl_score=old_toefl_score
    )


# ==========================================================
# TOEFL Report Generation
# ==========================================================

def toefl_report_from_evaluated_pairs(
    pairs: List[ToeflPairEvaluationResult],
) -> ToeflReport:
    """
    Generate a TOEFL report from a list of ToeflPairEvaluationResult objects.
    """
    valid_pairs = [p for p in pairs if not p.is_error]
    listen_repeat_pairs = [
        p for p in valid_pairs if p.pair_type == ToeflPairType.LISTEN_REPEAT
    ]
    interview_pairs = [p for p in valid_pairs if p.pair_type == ToeflPairType.INTERVIEW]

    errors = [
        PairError(pair_type=p.pair_type, idx=p.idx, error_message=p.error_message)
        for p in pairs if p.is_error
    ]
    generation_failed = len(valid_pairs) == 0

    listen_repeat_result = (
        _process_listen_repeat_pairs(listen_repeat_pairs)
        if listen_repeat_pairs else None
    )
    interview_result = (
        _process_interview_pairs(interview_pairs)
        if interview_pairs else None
    )

    # Weighted formula (1/3 listening + 2/3 interview)
    overall_score = None
    if listen_repeat_result and interview_result:
        weighted_speech_rate = int(
            round(
                ((1 / 3) * listen_repeat_result.speech_rate)
                + ((2 / 3) * interview_result.speech_rate)
            )
        )
        weighted_rep_accuracy = int(
            round(
                ((1 / 3) * listen_repeat_result.avg_rep_accuracy)
                + ((2 / 3) * interview_result.avg_rep_accuracy)
            )
        )
        overall_score = _calculate_overall_score(weighted_speech_rate, weighted_rep_accuracy)
    elif listen_repeat_result:
        overall_score = listen_repeat_result.overall_score
    elif interview_result:
        overall_score = interview_result.overall_score
    else:
        raise ValueError("No valid results available")

    return ToeflReport(
        generation_failed=generation_failed,
        errors=errors,
        overall_score=overall_score,
        listen_repeat_result=listen_repeat_result,
        interview_result=interview_result,
    )


# ==========================================================
# PART 1: Listen-Repeat Processing
# ==========================================================

def _process_listen_repeat_pairs(
    pairs: List[ToeflPairEvaluationResult],
) -> ListenRepeatResult:
    total_duration = sum(p.answer_audio_duration for p in pairs)
    total_words = sum(
        len(p.answer_audio_asr_text.split()) for p in pairs if p.answer_audio_asr_text
    )
    speech_rate = (
        int(round(total_words / (total_duration / 60.0))) if total_duration > 0 else 0
    )

    per_pair_fluency_list = []
    for p in pairs:
        total_words = len(p.answer_audio_asr_text.split())
        speech_rate = (
            int(round(total_words / (p.answer_audio_duration / 60.0)))
            if p.answer_audio_duration > 0 else 0
        )
        per_pair_fluency_list.append(
            ListenRepeatPerPairFluency(
                idx=p.idx,
                transcript=p.answer_audio_asr_text,
                speech_rate=speech_rate,
                duration=p.answer_audio_duration,
                accuracy=p.repeat_accuracy,
            )
        )

    rep_accuracy_scores = [p.repeat_accuracy for p in pairs if p.repeat_accuracy is not None]
    avg_rep_accuracy = int(round(sum(rep_accuracy_scores) / len(rep_accuracy_scores))) if rep_accuracy_scores else 0

    total_pauses = sum(p.pause_stats.pauses for p in pairs if p.pause_stats)
    total_long_pauses = sum(p.pause_stats.long_pauses for p in pairs if p.pause_stats)
    pauses_per_min = total_pauses / (total_duration / 60.0) if total_duration > 0 else 0
    long_ratio = total_long_pauses / total_pauses if total_pauses > 0 else 0

    all_mispronunciations = []
    for p in pairs:
        all_mispronunciations.extend(p.mispronunciations or [])
    all_mispro_words = [m.word for m in all_mispronunciations]

    pronunciation_accuracy_pct = _pronunciation_accuracy_pct(
        total_words, all_mispro_words
    )

    speech_rate_level = _level_from_speech_rate(speech_rate)
    coherence_level = _level_from_accuracy(avg_rep_accuracy)
    pause_frequency_level = _pause_frequency_level(pauses_per_min)
    pause_appropriateness_level = _pause_appropriateness_level(long_ratio)
    prosody_like_score = _prosody_like_score(pause_frequency_level)
    intonation_score = _intonation_like_score(speech_rate)

    overall_score = _calculate_overall_score(speech_rate, avg_rep_accuracy)

    fluency_summary, fluency_summary_cn = get_fluency_summary(
        speech_rate,
        pause_frequency_level,
        pause_appropriateness_level,
        avg_rep_accuracy,
    )
    pronunciation_summary, pronunciation_summary_cn = get_pronunciation_summary(
        all_mispro_words, pronunciation_accuracy_pct, total_words
    )
    grammar_summary, grammar_summary_cn = get_grammar_summary(
        issues=[], fallback_acc_pct=avg_rep_accuracy
    )

    return ListenRepeatResult(
        overall_score=overall_score,
        fluency=ListenRepeatFluency(
            speech_rate_score=_cefr_to_toefl(speech_rate_level),
            coherence_score=_cefr_to_toefl(coherence_level),
            pause_frequency_score=_cefr_to_toefl(pause_frequency_level),
            pause_appropriateness_score=_cefr_to_toefl(pause_appropriateness_level),
            repeat_accuracy_score=_cefr_to_toefl(coherence_level),
            per_pair_fluency=per_pair_fluency_list,
            summary=fluency_summary,
            summary_cn=fluency_summary_cn,
        ),
        pronunciation=Pronunciation(
            prosody_rhythm_score=prosody_like_score,
            vowel_fullness_score=_vowel_fullness_score(total_words, all_mispronunciations),  # ✅ [TODO: to be replaced with vowel fullness metric]
            pronunciation_and_intonation_score=intonation_score,
            summary=pronunciation_summary,
            summary_cn=pronunciation_summary_cn,
        ),
        mispronounced_words=all_mispronunciations,
        grammar=ListenRepeatGrammar(
            accuracy_score=_cefr_to_toefl(coherence_level),
            summary=grammar_summary,
            summary_cn=grammar_summary_cn,
        ),
        speech_rate=speech_rate,
        avg_rep_accuracy=avg_rep_accuracy,
    )


# ==========================================================
# PART 2: Interview Processing
# ==========================================================

def _process_interview_pairs(pairs: List[ToeflPairEvaluationResult]) -> InterviewResult:
    total_duration = sum(p.answer_audio_duration for p in pairs)
    duration_score = _calculate_duration_score(total_duration)  # ✅ [CHANGE: added helper below]

    total_words = sum(
        len(p.answer_audio_asr_text.split()) for p in pairs if p.answer_audio_asr_text
    )
    speech_rate = (
        int(round(total_words / (total_duration / 60.0))) if total_duration > 0 else 0
    )
    all_text = " ".join(
        p.answer_audio_asr_text for p in pairs if p.answer_audio_asr_text
    )

    per_pair_fluency_list = []
    for p in pairs:
        total_words = len(p.answer_audio_asr_text.split())
        rate = (
            int(round(total_words / (p.answer_audio_duration / 60.0)))
            if p.answer_audio_duration > 0 else 0
        )
        per_pair_fluency_list.append(
            PerPairFluency(
                idx=p.idx,
                speech_rate=rate,
                duration=p.answer_audio_duration,
                transcript=p.answer_audio_asr_text,
                accuracy=p.repeat_accuracy,
            )
        )

    rep_accuracy_scores = [p.repeat_accuracy for p in pairs if p.repeat_accuracy is not None]
    avg_rep_accuracy = int(round(sum(rep_accuracy_scores) / len(rep_accuracy_scores))) if rep_accuracy_scores else 0

    total_pauses = sum(p.pause_stats.pauses for p in pairs if p.pause_stats)
    total_long_pauses = sum(p.pause_stats.long_pauses for p in pairs if p.pause_stats)
    pauses_per_min = total_pauses / (total_duration / 60.0) if total_duration > 0 else 0
    long_ratio = total_long_pauses / total_pauses if total_pauses > 0 else 0

    all_mispronunciations = []
    for p in pairs:
        all_mispronunciations.extend(p.mispronunciations or [])
    all_mispro_words = [m.word for m in all_mispronunciations]
    pronunciation_accuracy_pct = _pronunciation_accuracy_pct(total_words, all_mispro_words)

    all_grammar_errors = []
    for p in pairs:
        all_grammar_errors.extend(p.grammar_feedback or [])

    speech_rate_level = _level_from_speech_rate(speech_rate)
    coherence_level = _level_from_accuracy(avg_rep_accuracy)
    pause_frequency_level = _pause_frequency_level(pauses_per_min)
    pause_appropriateness_level = _pause_appropriateness_level(long_ratio)
    word_repetition_level = _word_repetition_level(all_text)
    prosody_like_score = _prosody_like_score(pause_frequency_level)
    intonation_score = _intonation_like_score(speech_rate)
    vocabulary = _calculate_vocabulary_metrics(all_text)
    prompt_texts = [p.prompt_audio_asr_text for p in pairs if p.prompt_audio_asr_text]
    answer_texts = [p.answer_audio_asr_text for p in pairs if p.answer_audio_asr_text]
    relevance_score = _cefr_to_toefl(_embedding_relevance(prompt_texts, answer_texts))
    discourse_score = _cefr_to_toefl(_gpt4o_discourse(all_text))

    overall_score = _calculate_overall_score(speech_rate, avg_rep_accuracy)

    fluency_summary, fluency_summary_cn = get_fluency_summary(
        speech_rate,
        pause_frequency_level,
        pause_appropriateness_level,
        avg_rep_accuracy,
        word_repetition_level,
    )
    pronunciation_summary, pronunciation_summary_cn = get_pronunciation_summary(
        all_mispro_words, pronunciation_accuracy_pct, total_words
    )
    grammar_summary, grammar_summary_cn = get_grammar_summary(
        issues=all_grammar_errors, fallback_acc_pct=avg_rep_accuracy
    )

    # ✅ [CHANGE: Added vocabulary & relevance summaries]
    vocabulary_summary, vocabulary_summary_cn = get_vocabulary_and_relevance_summary(
        vocabulary, relevance_score, discourse_score
    )

    return InterviewResult(
        overall_score=overall_score,
        fluency=InterviewFluency(
            speech_rate_score=_cefr_to_toefl(speech_rate_level),
            coherence_score=_cefr_to_toefl(coherence_level),
            pause_frequency_score=_cefr_to_toefl(pause_frequency_level),
            pause_appropriateness_score=_cefr_to_toefl(pause_appropriateness_level),
            duration_score=duration_score,
            word_repetition_score=_cefr_to_toefl(word_repetition_level),
            per_pair_fluency=per_pair_fluency_list,
            summary=fluency_summary,
            summary_cn=fluency_summary_cn,
        ),
        pronunciation=Pronunciation(
            prosody_rhythm_score=prosody_like_score,
            vowel_fullness_score=_vowel_fullness_score(total_words, all_mispronunciations),
            pronunciation_and_intonation_score=intonation_score,
            summary=pronunciation_summary,
            summary_cn=pronunciation_summary_cn,
        ),
        mispronounced_words=all_mispronunciations,
        grammar=InterviewGrammar(
            accuracy_score=_cefr_to_toefl(coherence_level),
            grammar_errors=all_grammar_errors,
            summary=grammar_summary,
            summary_cn=grammar_summary_cn,
        ),
        vocabulary_and_relevance=VocabularyAndRelevance(
            relevance_score=relevance_score,
            discourse_score=discourse_score,
            vocabulary=vocabulary,
            summary=vocabulary_summary,
            summary_cn=vocabulary_summary_cn,
        ),
        speech_rate=speech_rate,
        avg_rep_accuracy=avg_rep_accuracy,
    )

### === PART 2/3 END ===


### === PART 3/3 START ===

# ==========================================================
# Fluency Utilities
# ==========================================================

def _repeat_accuracy_and_incorrect_segments(
    prompt_text: str, student_text: str
) -> Tuple[int, List[str]]:
    """Compute WER-based repeat accuracy and return incorrect spans."""
    w = wer(prompt_text.lower(), student_text.lower())
    score = max(0, int(round(100 * (1.0 - w))))
    p_tokens = prompt_text.split()
    s_tokens = student_text.split()
    sm = difflib.SequenceMatcher(a=p_tokens, b=s_tokens)
    incorrect = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("delete", "replace"):
            seg = " ".join(p_tokens[i1:i2]).strip()
            if seg:
                incorrect.append(seg)
    return score, incorrect


def _level_from_speech_rate(wpm: int) -> str:
    if wpm < 90:
        return "A1"
    if wpm < 110:
        return "A2"
    if wpm < 130:
        return "B1"
    if wpm < 150:
        return "B2"
    return "C1"


def _level_from_accuracy(pct: int) -> str:
    if pct < 50:
        return "A1"
    if pct < 70:
        return "A2"
    if pct < 85:
        return "B1"
    if pct < 95:
        return "B2"
    return "C1"


def _pause_frequency_level(ppm: float) -> str:
    if ppm > 20:
        return "A1"
    if ppm > 15:
        return "A2"
    if ppm > 10:
        return "B1"
    if ppm > 5:
        return "B2"
    return "C1"


def _pause_appropriateness_level(long_ratio: float) -> str:
    if long_ratio > 0.40:
        return "A1"
    if long_ratio > 0.30:
        return "A2"
    if long_ratio > 0.20:
        return "B1"
    if long_ratio > 0.10:
        return "B2"
    return "C1"


def _word_repetition_level(text: str) -> str:
    toks = [t.lower() for t in text.split() if t.isalpha()]
    if not toks:
        return "C1"
    from collections import Counter
    cnt = Counter(toks)
    rep_rate = sum((c - 1) for c in cnt.values() if c > 1) / len(toks)
    if rep_rate > 0.25:
        return "A1"
    if rep_rate > 0.18:
        return "A2"
    if rep_rate > 0.12:
        return "B1"
    if rep_rate > 0.07:
        return "B2"
    return "C1"


# ==========================================================
# Pronunciation
# ==========================================================

def _pronunciation_accuracy_pct(total_words: int, mis_words: list[str]) -> int:
    if total_words <= 0:
        return 100 if not mis_words else 0
    pct = int(round(100 * (1.0 - len(mis_words) / float(total_words))))
    return max(0, min(100, pct))


def _prosody_like_score(pause_freq_lvl: str) -> int:
    mapping = {"A1": "A1", "A2": "A2", "B1": "B2", "B2": "C1", "C1": "C1", "C2": "C2"}
    return _cefr_to_toefl(mapping.get(pause_freq_lvl, "A1"))


def _intonation_like_score(speech_rate_wpm: int) -> float:
    if speech_rate_wpm < 90:
        return 2.0
    if speech_rate_wpm < 110:
        return 3.0
    if speech_rate_wpm < 130:
        return 4.0
    if speech_rate_wpm < 150:
        return 5.0
    return 6.0


# ✅ [CHANGE: Added helper for vowel fullness score]
def _vowel_fullness_score(total_words: int, mispronunciations: List[Mispronunciations]) -> float:
    """
    Heuristic placeholder: penalize vowel fullness based on number of mispronounced words.
    """
    if total_words <= 0:
        return 1.0
    ratio = len(mispronunciations) / total_words
    if ratio > 0.25:
        return 1.0
    if ratio > 0.15:
        return 2.0
    if ratio > 0.10:
        return 3.0
    if ratio > 0.05:
        return 4.0
    return 5.0


# ==========================================================
# Vocabulary and Duration
# ==========================================================

def _calculate_duration_score(duration_sec: float) -> float:
    """Approximate adequacy of speech duration."""
    if duration_sec < 30:
        return 1.0
    if duration_sec < 45:
        return 2.0
    if duration_sec < 60:
        return 3.0
    if duration_sec < 75:
        return 4.0
    return 5.0


def _calculate_vocabulary_metrics(text: str) -> Vocabulary:
    if not text.strip():
        return Vocabulary(complexity_level="A1", diversity_level="A1")

    words = [w.lower() for w in text.split() if w.isalpha()]
    if not words:
        return Vocabulary(complexity_level="A1", diversity_level="A1")

    avg_length = sum(len(w) for w in words) / len(words)
    if avg_length < 4.2:
        complexity = "A1"
    elif avg_length < 4.8:
        complexity = "A2"
    elif avg_length < 5.4:
        complexity = "B1"
    elif avg_length < 6.0:
        complexity = "B2"
    else:
        complexity = "C1"

    unique_words = len(set(words))
    ttr = unique_words / len(words)
    if ttr < 0.35:
        diversity = "A1"
    elif ttr < 0.45:
        diversity = "A2"
    elif ttr < 0.55:
        diversity = "B1"
    elif ttr < 0.65:
        diversity = "B2"
    else:
        diversity = "C1"

    return Vocabulary(
        complexity_score=_cefr_to_toefl(complexity),
        diversity_score=_cefr_to_toefl(diversity),
    )


# ✅ [CHANGE: Added missing vocabulary summary helper]
def get_vocabulary_and_relevance_summary(vocab: Vocabulary, rel: float, disc: float):
    """
    Produce natural-language summary for vocabulary + relevance section.
    """
    summary = f"Vocabulary shows complexity around CEFR {vocab.complexity_score} and diversity {vocab.diversity_score}. "
    summary += f"Semantic relevance score {rel:.2f} and discourse score {disc:.2f} suggest coherent topic maintenance."
    summary_cn = "词汇复杂度与多样性均衡，语义相关性与篇章连贯性良好。"
    return summary, summary_cn


# ==========================================================
# Embedding Relevance
# ==========================================================

def _embedding_relevance(prompt_texts: List[str], answer_texts: List[str]) -> str:
    pairs = [
        (q.strip(), a.strip())
        for q, a in zip(prompt_texts, answer_texts)
        if q.strip() and a.strip()
    ]
    if not pairs:
        return "A1"

    try:
        client = OpenAI()
        texts = []
        for q, a in pairs:
            texts.append(q)
            texts.append(a)
        embs = client.embeddings.create(
            model="text-embedding-3-small", input=texts
        ).data
        vecs = [e.embedding for e in embs]
        labels = []
        for i in range(0, len(vecs), 2):
            qv, av = vecs[i], vecs[i + 1]
            sim = _cosine(qv, av)
            labels.append(_sim_to_cefr(sim))
        order = ["A1", "A2", "B1", "B2", "C1", "C2"]
        return min(labels, key=order.index) if labels else "A1"
    except Exception:
        return "B1" if len(pairs) >= 2 else "A2"


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    vec_a = np.array(a)
    vec_b = np.array(b)
    dot = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def _sim_to_cefr(sim: float) -> str:
    if sim >= 0.82:
        return "C1"
    if sim >= 0.74:
        return "B2"
    if sim >= 0.66:
        return "B1"
    if sim >= 0.58:
        return "A2"
    return "A1"


# ==========================================================
# Discourse (GPT-4o)
# ==========================================================

def _gpt4o_discourse(answer_text: str) -> str:
    if not answer_text.strip():
        return "A1"
    try:
        client = OpenAI()
        system = {
            "role": "system",
            "content": "Rate discourse coherence and structure from CEFR A1–C1.",
        }
        user = {"role": "user", "content": f"Text:\n{answer_text}"}
        resp = client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[system, user],
            temperature=0,
            response_format=DiscourseEvaluation,
        )
        parsed = resp.choices[0].message.parsed
        return parsed.label if parsed else "A1"
    except Exception:
        words = answer_text.split()
        if len(words) < 10:
            return "A1"
        elif len(words) < 20:
            return "A2"
        elif len(words) < 50:
            return "B1"
        elif len(words) < 100:
            return "B2"
        else:
            return "C1"


# ==========================================================
# CEFR → TOEFL numeric mapping
# ==========================================================

def _cefr_to_toefl(lab: str) -> float:
    return {"A1": 1.0, "A2": 2.0, "B1": 3.0, "B2": 4.0, "C1": 5.0, "C2": 6.0}.get(
        (lab or "").upper(), 1.0
    )

### === PART 3/3 END ===
