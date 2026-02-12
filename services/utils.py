import torch
import random
from transformers import (
    BlenderbotTokenizer, 
    BlenderbotForConditionalGeneration,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    AutoTokenizer,
    AutoModelForSequenceClassification
)
import torch.nn.functional as F
import os

# Force CPU usage (more stable for Mac MPS issues)
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
device = torch.device("cpu")
print(f"🖥️  Using device: {device}")

# Emotion mappings
emotions = [
    'jealous', 'furious', 'disgusted', 'nostalgic', 'impressed', 'faithful',
    'caring', 'confident', 'guilty', 'angry', 'disappointed', 'sentimental',
    'anxious', 'annoyed', 'embarrassed', 'terrified', 'apprehensive', 'grateful',
    'sad', 'afraid', 'ashamed', 'devastated', 'joyful', 'hopeful', 'lonely',
    'prepared', 'trusting', 'anticipating', 'excited', 'surprised', 'content', 'proud'
]
emotion_to_id = {emotion: idx for idx, emotion in enumerate(emotions)}
id_to_emotion = {idx: emotion for emotion, idx in emotion_to_id.items()}

# Safety responses
MILD_RESPONSES = [
    "💙 It sounds like you're going through a tough time. You're not alone.",
    "🫶 I'm really sorry you're feeling this way. Please know that help is available.",
    "🌻 You matter. Please reach out to someone you trust or a professional.",
    "🌸 I'm here for you. Talking to a counselor can really help in moments like these."
]

EXTREME_RESPONSES = [
    "🚨 I'm deeply concerned about your safety. Please talk to a mental health professional or call a crisis hotline immediately.",
    "⚠️ It sounds like you're in a lot of pain. I'm not a crisis service, but you're not alone — please reach out to a counselor or crisis line now.",
    "⛑️ I'm just a support tool and not equipped to help in a crisis. Please talk to a licensed mental health professional right away."
]

HIGH_SEVERITY_KEYWORDS = {
    "kill myself", "end of me", "want to die", "suicide", "die", 
    "can't go on", "ending it all", "not worth living"
}

# Global model variables (loaded once)
blender_model = None
blender_tokenizer = None
emotion_model = None
emotion_tokenizer = None
safety_model = None
safety_tokenizer = None


def load_models(model_path=None):
    if model_path is None:
        model_path = os.getenv("MODEL_PATH", "./models")
    """Load all models once at startup"""
    global blender_model, blender_tokenizer
    global emotion_model, emotion_tokenizer
    global safety_model, safety_tokenizer
    
    print("Loading BlenderBot model...")
    blender_tokenizer = BlenderbotTokenizer.from_pretrained(f"{model_path}/blender_empathetic_final/")
    blender_model = BlenderbotForConditionalGeneration.from_pretrained(f"{model_path}/blender_empathetic_final/")
    blender_model.to(device)
    
    print("Loading emotion detection model...")
    emotion_tokenizer = RobertaTokenizer.from_pretrained(f"{model_path}/rob-large-emotion-detector_dedupe/")
    emotion_model = RobertaForSequenceClassification.from_pretrained(f"{model_path}/rob-large-emotion-detector_dedupe/")
    emotion_model.to(device)
    
    print("Loading safety detection model...")
    safety_tokenizer = AutoTokenizer.from_pretrained("sentinet/suicidality")
    safety_model = AutoModelForSequenceClassification.from_pretrained("sentinet/suicidality")
    safety_model.to(device)
    
    print("✅ All models loaded successfully!")


def detect_emotion(text: str) -> str:
    """Detect emotion from text"""
    inputs = emotion_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    ).to(device)
    
    with torch.no_grad():
        logits = emotion_model(**inputs).logits
    probs = F.softmax(logits, dim=-1)
    idx = probs.argmax(dim=-1).item()
    
    return id_to_emotion[idx]


def detect_distress_and_severity(text: str, model_threshold=0.7) -> str:
    """Detect if user is in distress and return severity level"""
    # Model-based detection
    inputs = safety_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = safety_model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    distress_score = probs[0][1].item()
    
    # Keyword-based detection
    text_lower = text.lower()
    high_severity_flag = any(phrase in text_lower for phrase in HIGH_SEVERITY_KEYWORDS)
    
    is_distressed = distress_score > model_threshold
    severity = "extreme" if high_severity_flag else "mild" if is_distressed else "none"
    
    return severity


def get_safety_response(severity: str) -> str:
    """Get appropriate safety response based on severity"""
    if severity == "extreme":
        return random.choice(EXTREME_RESPONSES)
    elif severity == "mild":
        return random.choice(MILD_RESPONSES)
    return None


def adjust_response(response: str, emotion: str) -> str:
    """Add emotion-appropriate prefix to response"""
    emotion_responses = {
        'furious': ["😡 That's infuriating! ", "💢 This is unacceptable! "],
        'proud': ["🏆 Incredible achievement! ", "👏 You should be proud! "],
        'nostalgic': ["🕰️ Reminiscing can be powerful. ", "📻 Those memories matter. "],
        'jealous': ["💚 It's natural to feel this way. ", "🤢 Jealousy is tough. "],
        'anticipating': ["⏳ The wait must be intense. ", "🔮 Exciting things ahead! "],
        'sentimental': ["📜 Those feelings are valid. ", "💌 Heartfelt moments. "],
        'grateful': ["🙏 Gratitude changes everything. ", "🌈 Appreciation is beautiful. "],
        'caring': ["💖 Your compassion shines. ", "🤗 Kindness matters. "],
        'hopeful': ["🌟 Hope fuels progress. ", "🔭 Looking forward with you. "],
        'devastated': ["💔 This is heartbreaking. ", "🕯️ I'm here in this pain. "],
        'terrified': ["😱 That sounds terrifying! ", "🛡️ Let's find safety. "],
        'ashamed': ["😞 These feelings are valid. ", "🛑 You're safe here. "],
    }
    
    default_prefixes = {
        'positive': "😊 ",
        'negative': "😟 ",
        'neutral': "🤖 "
    }
    
    if emotion in emotion_responses:
        prefix = random.choice(emotion_responses[emotion])
    else:
        if emotion in ['joyful', 'excited', 'confident']:
            prefix = default_prefixes['positive']
        elif emotion in ['sad', 'anxious', 'guilty']:
            prefix = default_prefixes['negative']
        else:
            prefix = default_prefixes['neutral']
    
    return f"{prefix}{response}"


def generate_response(message: str, conversation_history: list = None) -> dict:
    """
    Main inference function - generates empathetic response
    
    Returns:
        dict with keys: response, emotion, safety_triggered
    """
    # 1. Check for distress first
    severity = detect_distress_and_severity(message)
    safety_response = get_safety_response(severity)
    
    if safety_response:
        return {
            "response": safety_response,
            "emotion": "concerned",
            "safety_triggered": True,
            "severity": severity
        }
    
    # 2. Detect emotion
    emotion = detect_emotion(message)
    
    # 3. Build input with context
    if conversation_history:
        context = " [SEP] ".join(conversation_history[-3:])  # Last 3 turns
        input_text = f"<emotion={emotion}> [CONTEXT] {context} [USER] {message}"
    else:
        input_text = f"<emotion={emotion}> [USER] {message}"
    
    # 4. Generate response
    inputs = blender_tokenizer(
        input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True
    ).to(device)
    
    outputs = blender_model.generate(
        **inputs,
        max_new_tokens=64,
        temperature=0.9,
        top_k=50,
        top_p=0.95,
        repetition_penalty=2.0,
        num_beams=4,
        do_sample=True,
        no_repeat_ngram_size=2,
        length_penalty=0.9,
        early_stopping=True
    )
    
    response = blender_tokenizer.decode(outputs[0], skip_special_tokens=True)
    adjusted_response = adjust_response(response, emotion)
    
    return {
        "response": adjusted_response,
        "emotion": emotion,
        "safety_triggered": False,
        "severity": "none"
    }