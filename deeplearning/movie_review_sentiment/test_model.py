
import torch

from custom_dataset import build_vocab, load_data
from models import CNNClassifier, LSTMClassifier, RNNClassifier, TransformerClassifier
from tokenizers import my_tokenizer_number

# function to predict sentiment of a single sentence using the trained model
def predict_sentence(model, sentence, vocab, max_len=200, device="cpu"):
    indices = list(my_tokenizer_number([sentence], vocab))
    if len(indices[0]) < max_len:
        indices[0] += [0] * (max_len - len(indices[0]))
    else:
        indices[0] = indices[0][:max_len]
    input_tensor = torch.tensor([indices[0]], dtype=torch.long).to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output)
        label = prob.round()
    return prob.item(), int(label.item())

# set constants
MAX_LEN = 200

# load data and build vocab
dataset_local = load_data(source='LOCAL')
# create vocab from training data only to prevent data leakage (like training process)
vocab = build_vocab(dataset_local['train'], text_column='text', vocab_size=25000, min_freq=2)

# choose model architecture and initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "transformer"  # change to "rnn","lstm", "cnn", "transformer"

model_name = "transformer"  # change to "rnn","lstm", "cnn", "transformer"
if model_name == "rnn":
    model = RNNClassifier(
        vocab_size=len(vocab),
        embed_dim=128,
        hidden_dim=256,
        num_classes=1,
        num_layers=1,
        dropout=0.4,
        bidirectional=True
    ).to(device)
elif model_name == "lstm":
    model = LSTMClassifier(
        vocab_size=len(vocab),
        embed_dim=128,
        hidden_dim=256,
        num_classes=1,
        num_layers=1,
        dropout=0.4,
        bidirectional=True
    ).to(device)
elif model_name == "cnn":
    model = CNNClassifier(
        vocab_size=len(vocab),
        embed_dim=128,
        num_classes=1,
        kernel_sizes=[3,4,5],
        num_filters=100,
        dropout=0.5
    ).to(device)
elif model_name == "transformer":
    model = TransformerClassifier(
        vocab_size=len(vocab),
        embed_dim=128,
        num_classes=1,
        num_heads=4,
        hidden_dim=256,
        num_layers=2,
        dropout=0.2,
        max_len=500
    ).to(device)



# Load best model
model.load_state_dict(torch.load("./results_rnn/best_model_by_loss.pth"))
# model.eval()

# sentence = "I loved this movie, it was great!"
# sentence = "It was not a good movie, I hated it. I think it was the worst film I've ever seen."
sentence = "It was not a good movie, I hated it."

# Usage:
score, label = predict_sentence(model, sentence, vocab, MAX_LEN, device)
print(f"Score={score:.4f}, Label={label}")
