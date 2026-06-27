from settings import (
    RERANK_BATCH_SIZE,
    RERANK_MAX_LENGTH,
    RERANK_MODEL,
    RERANK_USE_FP16,
)


class BGEReranker:
    def __init__(
        self,
        model_name: str,
        use_fp16: bool = False,
        batch_size: int = 4,
        max_length: int = 512,
    ):
        try:
            import torch
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise RuntimeError(
                "BAAI reranking requires torch and transformers. "
                "Install requirements.txt first."
            ) from exc

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.torch = torch
        self.batch_size = batch_size
        self.max_length = max_length
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                local_files_only=True,
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                local_files_only=True,
            )
        except OSError:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        if use_fp16 and self.device.type != "cpu":
            self.model = self.model.half()
        self.model = self.model.to(self.device)
        self.model.eval()

    def compute_score(self, pairs, batch_size=None, max_length=None):
        if not pairs:
            return []
        batch_size = batch_size or self.batch_size
        max_length = max_length or self.max_length
        scores = []

        with self.torch.inference_mode():
            for start in range(0, len(pairs), batch_size):
                batch = pairs[start : start + batch_size]
                encoded = self.tokenizer(
                    [pair[0] for pair in batch],
                    [pair[1] for pair in batch],
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                encoded = {
                    key: value.to(self.device)
                    for key, value in encoded.items()
                }
                logits = self.model(**encoded).logits.view(-1).float()
                scores.extend(logits.cpu().tolist())
        return scores


def load_reranker():
    return BGEReranker(
        RERANK_MODEL,
        use_fp16=RERANK_USE_FP16,
        batch_size=RERANK_BATCH_SIZE,
        max_length=RERANK_MAX_LENGTH,
    )


def rerank(query, candidates, ranker, top_k=6):
    if not candidates:
        return []

    pairs = [[query, candidate["text"]] for candidate in candidates]
    scores = ranker.compute_score(
        pairs,
        batch_size=RERANK_BATCH_SIZE,
        max_length=RERANK_MAX_LENGTH,
    )
    if isinstance(scores, (int, float)):
        scores = [scores]

    ranked = sorted(
        zip(candidates, scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    return [
        {
            "id": candidate["id"],
            "text": candidate["text"],
            "metadata": candidate["metadata"],
            "score": float(score),
        }
        for candidate, score in ranked[:top_k]
    ]
