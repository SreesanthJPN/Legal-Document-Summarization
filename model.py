import torch.nn as nn
import torch

class ExtractiveSummarizer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout, n_heads=8):
        super(ExtractiveSummarizer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim * 2, num_heads=n_heads)
        
        self.conv1d = nn.Conv1d(in_channels=hidden_dim * 2, out_channels=hidden_dim, kernel_size=3, padding=1)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask):
        embedded = self.dropout(self.embedding(input_ids)) 
        lstm_output, _ = self.lstm(embedded)  
        lstm_output = self.layer_norm(lstm_output)
        
        lstm_output = lstm_output.permute(1, 0, 2)   
        attn_output, _ = self.attention(lstm_output, lstm_output, lstm_output)  
        attn_output = attn_output.permute(1, 0, 2)  
        
        conv_input = attn_output.permute(0, 2, 1) 
        conv_output = self.conv1d(conv_input)
        conv_output = conv_output.permute(0, 2, 1) 

        pooled_output = torch.mean(conv_output, dim=1)  

        output = self.fc(pooled_output)   
        
        return torch.sigmoid(output).squeeze(1)  