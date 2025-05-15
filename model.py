import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet101, ResNet101_Weights
from typing import List, Tuple, Dict

class ImageEncoder(nn.Module):
    """
    Image Encoder using ResNet101
    """
    def __init__(self, embed_dim: int = 2048, train_cnn: bool = False):
        """
        Args:
            embed_dim: Feature dimension
            train_cnn: Whether to train the CNN
        """
        super(ImageEncoder, self).__init__()

        # Load pre-trained ResNet101
        resnet = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)

        # Linear layer to project features to embed_dim
        self.linear = nn.Linear(resnet.fc.in_features, embed_dim)
        self.bn = nn.BatchNorm1d(embed_dim)
        self.dropout = nn.Dropout(0.5)

        # Freeze ResNet if not training
        if not train_cnn:
            for param in self.resnet.parameters():
                param.requires_grad = False
    
    def reset_parameters(self):
        """Reset the model parameters"""
        # Reset linear layer
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)
        # Reset batch norm
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract image features

        Args:
            images: Images of shape (batch_size, 3, height, width)

        Returns:
            features: Image features of shape (batch_size, embed_dim)
        """
        with torch.set_grad_enabled(self.training):
            features = self.resnet(images)

        features = features.reshape(features.size(0), -1)
        features = self.linear(features)
        features = self.bn(features)
        features = self.dropout(features)

        return features

class TopDownAttention(nn.Module):
    """
    Simplified Top-Down Attention for global image features (ResNet output)
    """
    def __init__(self,
                 embed_dim: int,
                 decoder_dim: int,
                 vocab_size: int,
                 word2idx: dict,
                 encoder_dim: int = 512,
                 dropout: float = 0.5):
        super().__init__()

        self.word2idx = word2idx
        self.encoder_dim = encoder_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout_layer = nn.Dropout(dropout)

        # Simplified LSTM (no region attention)
        self.attention_lstm = nn.LSTMCell(encoder_dim + embed_dim, decoder_dim)
        self.language_lstm = nn.LSTMCell(decoder_dim, decoder_dim)

        # Output layer
        self.fc = nn.Linear(decoder_dim, vocab_size)

        # Initialize weights
        self.init_weights()
    
    def reset_parameters(self):
        """Reset the model parameters"""
        self.init_weights()
        # Reset LSTM cells
        nn.init.xavier_uniform_(self.attention_lstm.weight_ih)
        nn.init.orthogonal_(self.attention_lstm.weight_hh)
        nn.init.constant_(self.attention_lstm.bias_ih, 0)
        nn.init.constant_(self.attention_lstm.bias_hh, 0)
        
        nn.init.xavier_uniform_(self.language_lstm.weight_ih)
        nn.init.orthogonal_(self.language_lstm.weight_hh)
        nn.init.constant_(self.language_lstm.bias_ih, 0)
        nn.init.constant_(self.language_lstm.bias_hh, 0)

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Args:
            encoder_out: (batch_size, encoder_dim) - ResNet global features
            encoded_captions: (batch_size, max_caption_length)
            caption_lengths: (batch_size,)
        """
        batch_size = encoder_out.size(0)

        # Sort by caption length (descending)
        caption_lengths, sort_ind = caption_lengths.sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embed captions
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_len, embed_dim)

        # Initialize LSTM states
        h_att = torch.zeros(batch_size, self.decoder_dim).to(encoder_out.device)
        c_att = torch.zeros(batch_size, self.decoder_dim).to(encoder_out.device)
        h_lang = torch.zeros(batch_size, self.decoder_dim).to(encoder_out.device)
        c_lang = torch.zeros(batch_size, self.decoder_dim).to(encoder_out.device)

        # Output tensors
        max_len = max(caption_lengths)
        predictions = torch.zeros(batch_size, max_len, self.vocab_size).to(encoder_out.device)

        for t in range(max_len):
            # Simplified input (no region features)
            att_lstm_input = torch.cat([embeddings[:, t, :], encoder_out], dim=1)
            h_att, c_att = self.attention_lstm(att_lstm_input, (h_att, c_att))

            # Language LSTM
            h_lang, c_lang = self.language_lstm(h_att, (h_lang, c_lang))

            # Prediction
            preds = self.fc(self.dropout_layer(h_lang))
            predictions[:, t, :] = preds

            # Early stop if all captions are processed
            if t >= caption_lengths[0]:
                break

        return predictions, None, encoded_captions, caption_lengths, sort_ind

    def sample(self, encoder_out, beam_size=5):
        """
        Generate a caption using beam search.

        Args:
            encoder_out: Encoded image features (1, encoder_dim)
            beam_size: Number of beams for beam search

        Returns:
            best_sequence: List of token indices representing the generated caption
        """
        k = beam_size
        vocab_size = self.vocab_size
        device = encoder_out.device

        # Expand for beam search
        encoder_out = encoder_out.expand(k, -1)

        # Initialize states
        h_att = torch.zeros(k, self.decoder_dim).to(device)
        c_att = torch.zeros(k, self.decoder_dim).to(device)
        h_lang = torch.zeros(k, self.decoder_dim).to(device)
        c_lang = torch.zeros(k, self.decoder_dim).to(device)

        # Start tokens
        start_tokens = torch.LongTensor([self.word2idx['<start>']] * k).to(device)
        embeddings = self.embedding(start_tokens)

        # Beam search setup
        seqs = start_tokens.unsqueeze(1)  # (k, 1)
        top_k_scores = torch.zeros(k, 1).to(device)
        complete_seqs = []
        complete_seqs_scores = []

        # Beam search loop
        for step in range(50):  # Max caption length
            # Simplified LSTM step
            lstm_input = torch.cat([embeddings, encoder_out], dim=1)
            h_att, c_att = self.attention_lstm(lstm_input, (h_att, c_att))
            h_lang, c_lang = self.language_lstm(h_att, (h_lang, c_lang))

            # Predict next word
            scores = self.fc(h_lang)  # (k, vocab_size)
            scores = F.log_softmax(scores, dim=1)  # (k, vocab_size)

            # Add current scores to cumulative scores
            scores = top_k_scores.expand_as(scores) + scores  # (k, vocab_size)

            # Flatten scores for top-k selection
            if step == 0:
                scores = scores[0]  # First step: only consider first sequence
                top_k_scores, top_k_words = scores.topk(k, dim=0)
                top_k_scores = top_k_scores.unsqueeze(1)  # (k, 1)
                prev_word_inds = torch.zeros(k, dtype=torch.long).to(device)  # (k,)
                next_word_inds = top_k_words  # (k,)
            else:
                scores = scores.view(-1)  # (k * vocab_size)
                top_k_scores, top_k_pos = scores.topk(k, dim=0)  # (k,), (k,)
                prev_word_inds = top_k_pos // vocab_size  # (k,)
                next_word_inds = top_k_pos % vocab_size  # (k,)
                top_k_scores = top_k_scores.unsqueeze(1)  # (k, 1)

            # Update sequences
            new_seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)

            # Check for completed sequences
            incomplete_inds = []
            for i, next_word in enumerate(next_word_inds):
                if next_word == self.word2idx['<end>']:
                    complete_seqs.append(new_seqs[i].tolist())
                    complete_seqs_scores.append(top_k_scores[i].item())
                else:
                    incomplete_inds.append(i)

            k = len(incomplete_inds)
            if k == 0:
                break

            # Update sequences and states for incomplete sequences
            seqs = new_seqs[incomplete_inds]
            h_att = h_att[prev_word_inds[incomplete_inds]]
            c_att = c_att[prev_word_inds[incomplete_inds]]
            h_lang = h_lang[prev_word_inds[incomplete_inds]]
            c_lang = c_lang[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds]
            
            # Update embeddings for next step
            embeddings = self.embedding(next_word_inds[incomplete_inds])

        # Select best sequence
        if complete_seqs:
            best_idx = complete_seqs_scores.index(max(complete_seqs_scores))
            best_sequence = complete_seqs[best_idx]
        else:
            # If no sequences completed, take the highest-scoring active sequence
            best_idx = top_k_scores.argmax().item()
            best_sequence = seqs[best_idx].tolist()

        return best_sequence
