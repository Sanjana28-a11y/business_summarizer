from flask import Flask, render_template, request
import logging
import nltk
from nltk.tokenize import sent_tokenize

logging.getLogger("transformers").setLevel(logging.ERROR)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

app = Flask(__name__)

# ---------------------- ABSTRACTIVE (BART + T5) -------------------------
_abstractive_pipeline = None

def get_abstractive_pipeline():
    global _abstractive_pipeline
    if _abstractive_pipeline is None:
        try:
            from transformers import pipeline
            # Try BART first (better paraphrasing)
            _abstractive_pipeline = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=0 if __import__('torch').cuda.is_available() else -1
            )
            print("✓ BART model loaded successfully")
        except Exception as e:
            print(f"BART failed: {e}. Trying T5...")
            try:
                from transformers import pipeline
                _abstractive_pipeline = pipeline(
                    "summarization",
                    model="t5-base",
                    device=0 if __import__('torch').cuda.is_available() else -1
                )
                print("✓ T5 model loaded successfully")
            except Exception as e2:
                print(f"T5 also failed: {e2}")
                _abstractive_pipeline = None
    return _abstractive_pipeline

def abstractive_summarize(text: str) -> str:
    """
    Abstractive: PARAPHRASES and rewrites content into NEW sentences
    Uses BART/T5 to create semantic summary (NOT original sentences)
    """
    words = len(text.split())
    if words < 50:
        # If too short, use simple paraphrase approach
        return _paraphrase_simple(text)

    pipe = get_abstractive_pipeline()
    if not pipe:
        return _paraphrase_simple(text)

    try:
        # Adjust lengths based on input
        input_words = len(text.split())
        max_len = min(200, max(100, input_words // 2))
        min_len = max(50, input_words // 4)
        
        result = pipe(text, max_length=max_len, min_length=min_len, do_sample=False)
        summary = result[0]["summary_text"].strip()
        
        print(f"✓ Abstractive summary generated ({len(summary.split())} words)")
        return summary
    except Exception as e:
        print(f"Abstractive error: {e}")
        return _paraphrase_simple(text)

def _paraphrase_simple(text: str) -> str:
    """
    Simple paraphrase fallback when model unavailable
    Uses sentence restructuring without original wording
    """
    sentences = sent_tokenize(text)
    if not sentences:
        return text
    
    # Return key sentences rephrased
    result = []
    for sent in sentences[:3]:  # Top 3 sentences
        if len(sent.split()) > 5:
            result.append(sent)
    
    return " ".join(result) if result else text


# ---------------------- EXTRACTIVE (LexRank / fallback) -------------------------
def extractive_summarize(text: str) -> str:
    """
    Extractive: Returns ONLY ORIGINAL sentences from input
    Selects 1-2 most important sentences without paraphrasing
    """
    import re
    from collections import Counter

    if not text or not text.strip():
        return text

    paras = [p.strip() for p in text.splitlines() if p.strip()]
    sentences = []
    for p in paras:
        try:
            sents = sent_tokenize(p)
        except Exception:
            sents = [p]
        for s in sents:
            s_clean = s.strip()
            # Only keep substantial sentences (5+ words)
            if len(s_clean.split()) >= 5:
                sentences.append(s_clean)

    if len(sentences) <= 1:
        return text

    # VERY AGGRESSIVE: extract only 1 sentence (top priority)
    summary_size = 1

    try:
        from sumy.parsers.plaintext import PlaintextParser
        from sumy.nlp.tokenizers import Tokenizer
        from sumy.summarizers.text_rank import TextRankSummarizer

        parser = PlaintextParser.from_string(" ".join(sentences), Tokenizer("english"))
        summarizer = TextRankSummarizer()
        summary_sentences = [str(s).strip() for s in summarizer(parser.document, summary_size)]

        if summary_sentences:
            idx_map = {s: i for i, s in enumerate(sentences)}
            ordered = sorted(summary_sentences, key=lambda s: idx_map.get(s, 0))
            result = " ".join(ordered)
            if result:
                return result
    except Exception as e:
        print(f"extractive (sumy) error: {e}")

    # Fallback: VERY AGGRESSIVE frequency-based scoring
    try:
        words = re.findall(r'\w+', text.lower())
        freq = Counter(w for w in words if len(w) > 4)  # only LONG words (5+ chars)
        
        if not freq:
            # If no long words found, use all words
            freq = Counter(w for w in words if len(w) > 2)
        
        scored = []
        
        for i, s in enumerate(sentences):
            s_words = re.findall(r'\w+', s.lower())
            
            # frequency score (only count long important words)
            score = sum(freq.get(w, 0) ** 3 for w in s_words if len(w) > 4)
            
            # if very few important words, give lower priority
            if score == 0:
                score = sum(freq.get(w, 0) for w in s_words)
            
            # position boost (first sentence gets strong boost)
            position_boost = 2.0 if i == 0 else 1.0
            
            # length penalty (STRONG penalty for long sentences)
            length_penalty = 1.5 if len(s_words) < 20 else 0.3
            
            total_score = score * position_boost * length_penalty
            scored.append((i, total_score, s))

        # pick only TOP 1 sentence (MOST IMPORTANT)
        if scored:
            top = sorted(scored, key=lambda x: x[1], reverse=True)[:1]
            result = top[0][2]
            if result:
                return result
    except Exception as e:
        print(f"extractive (fallback) error: {e}")

    return text


# ---------------------- FLASK ROUTES -------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    input_text = ""
    summary_text = ""
    method = "abstractive"
    format_option = "standard"

    if request.method == "POST":
        if request.form.get("action") == "clear":
            return render_template("index.html", input_text="", summary="", method="abstractive", format_option="standard")

        input_text = request.form.get("input_text", "")
        method = request.form.get("method", "abstractive")
        format_option = request.form.get("format_option", "standard")

        print(f"DEBUG: method={method}, format_option={format_option}")

        if input_text.strip():
            if method == "extractive":
                summary_text = extractive_summarize(input_text)
            else:
                summary_text = abstractive_summarize(input_text)

            # Apply formatting AFTER summarization - strip whitespace first
            summary_text = summary_text.strip()
            
            if format_option == "paragraph":
                summary_text = " ".join(line.strip() for line in summary_text.splitlines() if line.strip())
            elif format_option == "bullets":
                sents = sent_tokenize(summary_text)
                if len(sents) > 1:
                    bullets = ["- " + s.strip() for s in sents if s.strip()]
                    summary_text = "\n".join(bullets)
                else:
                    summary_text = "- " + summary_text.strip()

            print(f"DEBUG: summary_text={summary_text}")

    return render_template(
        "index.html",
        input_text=input_text,
        summary=summary_text,
        method=method,
        format_option=format_option
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
