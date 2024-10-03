import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

print("This is a test")


def get_device():
    """Determine whether to use GPU or CPU."""

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if hasattr(torch, "hip") and torch.hip.is_available():
        return torch.device("hip")
    return torch.device("cpu")


class RelevanceModel:
    """Check how relevant a switch review is"""

    def __init__(self):
        # model_name = "distilroberta-base"
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model_name = "microsoft/deberta-v3-large"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=1
        )
        self.device = get_device()
        print(f"Using device: {self.device}")
        self.model.to(self.device)

        self.characteristics = characteristics

        self.sentiment_words = sentiment_words

        self.switch_related = switch_related

    def predict_relevance(self, review: str) -> float:
        """Run Model against review"""
        prompt = f"""
        Evaluate the following review for a mechanical keyboard switch. A good review should:
        1. Discuss specific characteristics (e.g., {', '.join(self.characteristics)}, etc.)
        2. Provide personal opinions using words like {', '.join(self.sentiment_words)}, etc.
        3. Include switch-related terms such as {', '.join(self.switch_related)}, etc.
        4. Include an overall assessment
        6. The review must be related to mechanical keyboard switches

        Rate the review from 0.0 to 10.0, where:
        10.0 = Excellent, detailed review covering multiple aspects
        7.0-9.9 = Good review with some detailed information
        4.0-6.9 = Average review with basic information
        1.0-3.9 = Poor review with minimal relevant information
        0.0-0.9 = Not a relevant review

        Review to evaluate: {review}

        Rating:
        """

        inputs = self.tokenizer(
            prompt, return_tensors="pt", max_length=512, truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        score = torch.sigmoid(outputs.logits).item()
        mapped_score = score * 10

        char_count = sum(review.lower().count(char) for char in self.characteristics)
        sentiment_count = sum(
            review.lower().count(word) for word in self.sentiment_words
        )
        switch_related_count = sum(
            review.lower().count(word) for word in self.switch_related
        )

        if switch_related_count == 0:
            mapped_score *= 0.4

        if char_count == 0 and sentiment_count == 0:
            mapped_score *= 0.5

        if len(review.split()) > 20:
            mapped_score = min(mapped_score * 1.1, 10)

        if char_count > 0 and switch_related_count > 0:
            mapped_score = max(mapped_score, 4.0 + min(char_count, 3))

        mapped_score += min(sentiment_count, 2)

        if len(review.split()) < 10 and (
            char_count > 0 or switch_related_count > 0 or sentiment_count > 0
        ):
            mapped_score = max(mapped_score, 4.0)

        return round(min(max(mapped_score, 0.0), 10.0), 1)


relevance_model = RelevanceModel()

characteristics = [
    "feel",
    "sound",
    "pressure",
    "speed",
    "weight",
    "tactile",
    "clicky",
    "linear",
    "force",
    "actuation",
    "smooth",
    "gritty",
    "scratchy",
    "consistent",
    "wobble",
    "mushy",
    "firm",
    "crisp",
    "grainy",
    "responsive",
    "stiff",
    "bouncy",
    "stable",
    "thocky",
    "clacky",
    "ping",
    "muted",
    "hollow",
    "loud",
    "quiet",
    "dampened",
    "high-pitched",
    "deep",
    "rattly",
    "satisfying",
    "resistance",
    "light",
    "heavy",
    "balanced",
    "soft",
    "bottom-out",
    "preload",
    "fast",
    "slow",
    "snappy",
    "sluggish",
    "quick actuation",
    "laggy",
    "delay",
    "lightweight",
    "hefty",
    "medium-weight",
    "balanced-weight",
    "feather-light",
    "heavy-handed",
    "bump",
    "pronounced bump",
    "subtle",
    "sharp",
    "feedback",
    "gradual",
    "sharp click",
    "audible",
    "noticeable click",
    "loud click",
    "smooth travel",
    "effortless",
    "fluid",
    "actuation force",
    "bottom-out force",
    "low-force",
    "high-force",
    "short actuation",
    "high actuation",
    "actuation distance",
    "actuation point",
    "force curve",
    "low actuation",
]

switch_related = [
    "switch",
    "keyboard",
    "key",
    "typing",
    "mechanical",
    "keycaps",
    "stem",
    "spring",
    "housing",
    "plate",
    "stabilizer",
    "hotswap",
    "PCB",
    "travel distance",
    "debounce",
    "pre-travel",
    "post-travel",
    "ergonomics",
    "accuracy",
    "input",
    "keystroke",
    "typing experience",
    "typing feel",
    "keypress",
    "input lag",
    "rollover",
    "ghosting",
    "membrane",
    "optical switch",
    "hall effect",
    "MX-style",
    "Alps",
    "Topre",
    "Cherry MX",
    "Gateron",
    "Kailh",
    "Outemu",
    "Romer-G",
    "Zealios",
    "Box switches",
    "Holy Panda",
    "Speed switches",
    "Silent switches",
]

sentiment_words = [
    "good",
    "great",
    "bad",
    "excellent",
    "poor",
    "amazing",
    "terrible",
    "awesome",
    "disappointing",
    "okay",
    "meh",
    "fantastic",
    "satisfying",
    "premium",
    "buttery",
    "smooth",
    "flawless",
    "responsive",
    "top-notch",
    "impressive",
    "comfortable",
    "durable",
    "refined",
    "precise",
    "pleasing",
    "well-made",
    "perfect",
    "high-quality",
    "decent",
    "average",
    "not bad",
    "standard",
    "usable",
    "passable",
    "okayish",
    "so-so",
    "fine",
    "mediocre",
    "underwhelming",
    "lackluster",
    "frustrating",
    "inconsistent",
    "subpar",
    "uncomfortable",
    "weak",
    "annoying",
    "sluggish",
    "unsatisfying",
    "noisy",
    "disappointing",
]
# reviews = [
#     "This is a great feeling switch! I like how fast and responsive it is. Love the sound of it.",
#     "These switches feel great! Nice tactile bump and not too loud. Perfect for office use.",
#     "Clicky and satisfying. The actuation force is just right for me. Highly recommend!",
#     "Meh, they're okay I guess.",
#     "I bought a new keyboard.",
#     "The weather is nice today.",
#     "I love pizza!",
#     "This switch is terrible. It's too loud and feels mushy. The actuation point is inconsistent and it's just not pleasant to type on at all.",
#     "Smooth linear feel with a slight bump at the bottom. Great for gaming and typing. The sound is a bit louder than I expected, but still acceptable for office use.",
#     "These switches feel amazing. The tactile bump is smooth but pronounced, giving excellent feedback while typing. The sound is a deep, satisfying thock without being too loud, and the weight feels just right for extended typing sessions.",
#     "The weather was really nice today. I went for a walk and enjoyed the sunshine. Highly recommend walking in good weather!",
#     "The Gateron Browns offer a great balance between tactility and smoothness. They’re perfect for both typing and gaming, and the subtle bump is ideal if you prefer something quieter than clicky switches.",
# ]
#
# for review in reviews:
#     score = relevance_model.predict_relevance(review)
#     print(f"Review: {review}\nScore: {score}\n")
