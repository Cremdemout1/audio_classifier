import torch
import torchaudio
import numpy as np
import pandas as pd
from pathlib import Path
import math

#hyperarameters for log-mel:
sample_rate1 = 16000
n_fft = 1024 # determines the resolution of frequency axis
hop_length = 256 # determines the resolution of time axis | normally 1/4 of n_fft
n_mels = 80 # number of mel filter banks | 80 is a sweet spot for emotion discernment | sums th energy in frequency bands of n_mels size to avoid spiking sound

class Head(torch.nn.Module):
    Wq: torch.nn.Parameter # Query weight (what am I looking for)
    Wk: torch.nn.Parameter # Key weight (what am I offering)
    Wv: torch.nn.Parameter # Value weight (what information do I carry)

    bq: torch.nn.Parameter
    bk: torch.nn.Parameter
    bv: torch.nn.Parameter

    def __init__(self, in_features: int, out_features: int):
        super().__init__() # for nn params
        limit = math.sqrt(6 / (in_features + out_features))
        self.Wq = torch.nn.Parameter(torch.empty(in_features, out_features).uniform_(-limit, limit))
        self.Wk = torch.nn.Parameter(torch.empty(in_features, out_features).uniform_(-limit, limit))
        self.Wv = torch.nn.Parameter(torch.empty(in_features, out_features).uniform_(-limit, limit))

        self.bq = torch.nn.Parameter(torch.zeros(out_features))
        self.bk = torch.nn.Parameter(torch.zeros(out_features))
        self.bv = torch.nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor: # self attention

        Q = x @ self.Wq + self.bq
        K = x @ self.Wk + self.bk
        V = x @ self.Wv + self.bv
        # must use transpose instead of T because tensors are 3D [batch, seq_len, d_head] and I only want to transpose the 2 last dimensions
        self_attention = torch.softmax(Q @ K.transpose(-2, -1) / math.sqrt(K.shape[-1]), dim=-1) @ V
        return self_attention
    
class SemanticPerceptron(torch.nn.Module):
    W1: torch.nn.Parameter # dimension (inputLayer, hiddenLayer)
    W2: torch.nn.Parameter # dimension (hiddenLayer, output)

    b1: torch.nn.Parameter
    b2: torch.nn.Parameter
    labels: list
    gelu: any
    # ativation function must be relu or gelu
    def __init__(self, in_features):
        super().__init__() # for nn params
        hiddenLayerSize = int(in_features * 1.5)
        self.labels = ['request_command', 'personal_statement', 'narrative', 'factual_statement']
        outputSize = len(self.labels)
        limit1 = math.sqrt(6 / (in_features + hiddenLayerSize))
        limit2 = math.sqrt(6 / (hiddenLayerSize + outputSize))
    

        self.W1 = torch.nn.Parameter(torch.empty(in_features, hiddenLayerSize).uniform_(-limit1, limit1))
        self.W2 = torch.nn.Parameter(torch.empty(hiddenLayerSize, outputSize).uniform_(-limit2, limit2))
        self.b1 = torch.nn.Parameter(torch.zeros(hiddenLayerSize))
        self.b2 = torch.nn.Parameter(torch.zeros(outputSize))

        self.gelu = torch.nn.GELU()

    def feed_forward(self, input: torch.Tensor) -> torch.Tensor:
        hidden_layer = self.gelu(input @ self.W1 + self.b1)
        output_layer = hidden_layer @ self.W2 + self.b2

        return output_layer
        

class EmotionPerceptron(torch.nn.Module):
    W1: torch.nn.Parameter # dimension (inputLayer, hiddenLayer)
    W2: torch.nn.Parameter # dimension (hiddenLayer, output)

    b1: torch.nn.Parameter
    b2: torch.nn.Parameter
    labels: list
    gelu: any
    # ativation function must be relu or gelu
    def __init__(self, in_features):
        super().__init__() # for nn params
        hiddenLayerSize = int(in_features * 1.5)
        self.labels = ['Neutral', 'Happy', 'Sad', 'Anger', 'Disgust', 'Fear', 'Surprised', 'Calm']
        outputSize = len(self.labels)
        limit1 = math.sqrt(6 / (in_features + hiddenLayerSize))
        limit2 = math.sqrt(6 / (hiddenLayerSize + outputSize))
        

        self.W1 = torch.nn.Parameter(torch.empty(in_features, hiddenLayerSize).uniform_(-limit1, limit1))
        self.W2 = torch.nn.Parameter(torch.empty(hiddenLayerSize, outputSize).uniform_(-limit2, limit2))
        self.b1 = torch.nn.Parameter(torch.zeros(hiddenLayerSize))
        self.b2 = torch.nn.Parameter(torch.zeros(outputSize))

        self.gelu = torch.nn.GELU()

    def feed_forward(self, input: torch.Tensor) -> torch.Tensor:
        hidden_layer = self.gelu(input @ self.W1 + self.b1)
        output_layer = hidden_layer @ self.W2 + self.b2
        return output_layer


class transformer(torch.nn.Module):
    learning_rate: int
    epochs: int
    inputs: torch.Tensor
    W1: torch.nn.Parameter
    W2: torch.nn.Parameter
    b1: torch.nn.Parameter
    b2: torch.nn.Parameter

    heads: torch.nn.ModuleList
    MLPSem: SemanticPerceptron
    MLPEmo: EmotionPerceptron
    layerNorm: any
    gelu: any
    crossEntropyLoss: any
    optimizer: any
    d_model: int
    d_head: int
    max_chunks: int
    mel_transform: any

    def __init__(self, learning_rate=0.1, epochs=1000, audios="audio.csv", numHeads=6):
        super().__init__() # for nn params
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.inputs = self.load_tensors_to_input(audios)

        inputs_flat = self.inputs.view(self.inputs.shape[0], self.inputs.shape[1], -1)  # [batch, num_chunks, d_model]
        # d_model = inputs_flat.shape[-1]
        d_model = self.inputs.view(self.inputs.shape[0], self.inputs.shape[1], -1).shape[-1]
        if d_model % numHeads != 0:
            new_d_model = numHeads * ((d_model // numHeads) + 1)
            pad = new_d_model - d_model
            self.inputs = torch.nn.functional.pad(self.inputs, (0, pad))
            d_model = new_d_model
        d_head = d_model // numHeads
        limit = math.sqrt(6 / (d_model + d_head))
        
        self.heads = torch.nn.ModuleList([Head(in_features=d_model, out_features=d_head) for _ in range(numHeads)])
        self.MLPSem = SemanticPerceptron(d_model)
        self.MLPEmo = EmotionPerceptron(d_model)
        hidden_dim = d_head * 4 # arbitrary
        self.W1 = torch.nn.Parameter(torch.empty(d_model, hidden_dim).uniform_(-limit, limit))
        self.W2 = torch.nn.Parameter(torch.empty(hidden_dim, d_model).uniform_(-limit, limit))
        self.b1 = torch.nn.Parameter(torch.zeros(hidden_dim))
        self.b2 = torch.nn.Parameter(torch.zeros(d_model))
        self.layerNorm = torch.nn.LayerNorm(d_model)
        self.gelu = torch.nn.GELU()
        self.crossEntropyLoss = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate) # must be ut after to take in all parameters
        self.d_model = d_model
        self.d_head = d_head
        self.mel_transform = mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate1,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )

    def load_tensors_to_input(self, csv_path: str) -> torch.Tensor:
        df = pd.read_csv(csv_path)
        
        tensors = []
        all_chunks = []
        max_chunks = 0

        for _, row in df.iterrows():
            path = Path(row["tensor_path"])
            tensor = torch.load(path)
            chunks = self.preprocess_tensor(tensor)
            all_chunks.append(chunks)
            max_chunks = max(max_chunks, chunks.shape[0])

        # pad all to max_chunks
        for chunks in all_chunks:
            pad_size = max_chunks - chunks.shape[0]
            if pad_size > 0:
                padding = torch.zeros((pad_size, chunks.shape[1], chunks.shape[2]))
                chunks = torch.cat([chunks, padding], dim=0)
            tensors.append(chunks)
        self.max_chunks = max_chunks
        return torch.stack(tensors)  # now shape: [batch, max_chunks, chunk_size, n_mels]

    def preprocess_tensor(self, tensor: torch.Tensor) -> torch.Tensor: # this function is designed to divide the audio in overlapped tensors to allow AI model to understand structure of sentence as well as sounds
        chunk_size = 24 # with log-mel, each frame becomesa 16ms. this means that if each chunk is 32 frames, each token(input) will be 512ms (0.5s) good for prosody but less good for phoneme
        # 60% overlap
        step = int(chunk_size * (1 - 0.6))
        # right now log_mel of tensor = [1, 80, T] where 1 is mono channel, 80 is the n_mels and T is the amount of timeeframes
        chunks = []
        T = tensor.shape[2]

        for start in range(0, T - chunk_size + 1, step):
            chunk = tensor[:, :, start:start + chunk_size]  # shape [1, n_mels, chunk_size]
            chunks.append(chunk)

        # pad last chunk if needed to reach chunk_size
        if T % step != 0 and T - chunk_size < 0:  # in case last chunk is smaller
            last_chunk = tensor[:, :, -chunk_size:]
            chunks.append(last_chunk)

        # manually stack (all chunks are now same size along n_mels and chunk_size)
        processed_tensor = torch.stack(chunks, dim=0)  # shape [num_chunks, 1, n_mels, chunk_size]
        return processed_tensor
    
    def preprocess_rec(self, waveform: torch.Tensor, chunk_size=24):
        # 1. Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        waveform = waveform.squeeze(-1)
        # 2. Mel-spectrogram
        mel_spec = self.mel_transform(waveform)  # [1, n_mels, T]
        
        # 3. Log-mel
        log_mel = torch.log1p(mel_spec)
        
        # 4. Pad along time axis to at least chunk_size
        if log_mel.shape[-1] < chunk_size:
            log_mel = torch.nn.functional.pad(log_mel, (0, chunk_size - log_mel.shape[-1]))
        
        # 5. Chunking
        step = int(chunk_size * (1 - 0.6))
        chunks = []
        T = log_mel.shape[-1]
        for start in range(0, T - chunk_size + 1, step):
            chunk = log_mel[:, :, start:start+chunk_size]  # [1, n_mels, chunk_size]
            chunks.append(chunk)
        if T % step != 0 and T - chunk_size < 0:  # pad last chunk
            last_chunk = log_mel[:, :, -chunk_size:]
            chunks.append(last_chunk)
        
        chunks_tensor = torch.stack(chunks, dim=0)  # [num_chunks, 1, n_mels, chunk_size]
        return chunks_tensor


    def predict(self, user_recording: torch.Tensor = None, input: int = 0,
            training: bool = False, targetSemantic: str = 'none', targetEmotion: str = 'none') -> list[str]:
        
        if user_recording is not None:
            chunks = self.preprocess_rec(user_recording) 

            x = chunks.view(chunks.shape[0], -1)
            if x.shape[1] < self.d_model:
                x = torch.nn.functional.pad(x, (0, self.d_model - x.shape[1]))
            elif x.shape[1] > self.d_model:
                x = x[:, :self.d_model]

            if x.shape[0] < self.max_chunks:
                pad = torch.zeros(self.max_chunks - x.shape[0], self.d_model)
                x = torch.cat([x, pad], dim=0)
            else:
                x = x[:self.max_chunks]
            # print(x)
            # print(
            # user_recording.min().item(),
            # user_recording.max().item(),
            # user_recording.abs().mean().item()
            # )
            # print(chunks.shape)
            # print(chunks.abs().mean(dim=(1,2,3)))

        else:
            x = self.inputs[input].view(self.inputs.shape[1], -1)

        # flatten per chunk to match d_model
        # reshape [num_chunks, 1, n_mels, chunk_size] -> [num_chunks, d_model]

        multihead_output = torch.cat([head.forward(x) for head in self.heads], dim=-1)
        
        X1 = self.layerNorm(x + multihead_output)
        FF = self.gelu(X1 @ self.W1 + self.b1)
        FF = FF @ self.W2 + self.b2
        X2 = self.layerNorm(X1 + FF)
        
        semantic_repr = X2.mean(dim=0, keepdim=True)   # [1, d_model]
        emotion_repr = X2.max(dim=0).values.unsqueeze(0)  # [1, d_model]
        
        semantic_logits = self.MLPSem.feed_forward(semantic_repr)
        emotion_logits = self.MLPEmo.feed_forward(emotion_repr)
        
        pred_sem = torch.argmax(semantic_logits, dim=-1).item()
        pred_emo = torch.argmax(emotion_logits, dim=-1).item()
        
        semantic = self.MLPSem.labels[pred_sem]
        emotion = self.MLPEmo.labels[pred_emo]
        
        if training:
            semantic_target_index = self.MLPSem.labels.index(targetSemantic)
            emotion_target_index = self.MLPEmo.labels.index(targetEmotion)
            semantic_targets = torch.tensor([semantic_target_index], dtype=torch.long)
            emotion_targets = torch.tensor([emotion_target_index], dtype=torch.long)
            
            loss_sem = self.crossEntropyLoss(semantic_logits, semantic_targets)
            loss_emo = self.crossEntropyLoss(emotion_logits, emotion_targets)
            total_loss = loss_sem + loss_emo
            print(f"Loss: {total_loss.item():.4f}")
            total_loss.backward()
        
        return [semantic, emotion]

    def make_batches(self, indexes, batch_size):
        for i in range(0, len(indexes), batch_size):
            yield indexes[i : i + batch_size]

    def fit(self):
        training_samples = []
        testing_samples = []

        batch_size = 16

        training_samples = np.random.randint(0, 8881 + 1, 8881)
        testing_samples = np.random.randint(0, 7441 + 1, 7442 - 4000)

        # length = len(training_samples)
        df = pd.read_csv('labels.csv')

        for _ in range(self.epochs):
            np.random.shuffle(training_samples)
            
            for batch_indexes in self.make_batches(training_samples, batch_size):
                x = self.inputs[batch_indexes]
                x = x.view(x.shape[0], x.shape[1], -1) # change tensor to shape [batch, seq_len, d_model]
                multihead_output = torch.cat([head(x) for head in self.heads], dim=-1)
        
                X1 = self.layerNorm(x + multihead_output)
                FF = self.gelu(X1 @ self.W1 + self.b1)
                FF = FF @ self.W2 + self.b2
                X2 = self.layerNorm(X1 + FF)
                
                semantic_repr = X2.mean(dim=1)   # [1, d_model]
                emotion_repr = X2.max(dim=1).values # [1, d_model]
                
                semantic_logits = self.MLPSem.feed_forward(semantic_repr)
                emotion_logits = self.MLPEmo.feed_forward(emotion_repr)

                semantic_targets = torch.tensor(
                    [self.MLPSem.labels.index(df.loc[i, "Semantic"]) for i in batch_indexes],
                    dtype=torch.long
                )
                emotion_targets = torch.tensor(
                    [self.MLPEmo.labels.index(df.loc[i, "Emotion"]) for i in batch_indexes],
                    dtype=torch.long
                )
                
                loss_sem = self.crossEntropyLoss(semantic_logits, semantic_targets)
                loss_emo = self.crossEntropyLoss(emotion_logits, emotion_targets)
                total_loss = loss_sem + loss_emo
                print(f"Loss: {total_loss.item():.4f}")
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                # if (semantic == semanticTarget):
                #     rightSemantic += 1
                # elif (emotion == emotionTarget):
                #     rightEmotion += 1
                # if (emotion == emotionTarget and semantic == semanticTarget):
                #     fullRight += 1
                # print(
                #     f"Sentence: {sentence}\n"
                #     f"Semantic -> Predicted: {semantic}, Expected: {semanticTarget}\n"
                #     f"Emotion  -> Predicted: {emotion}, Expected: {emotionTarget}\n"
                #     f"{'-'*50}"
                # )
                # print(
                # f"Emotion Accuracy: {rightEmotion/length*100:.2f}% | "
                # f"Semantic Accuracy: {rightSemantic/length*100:.2f}% | "
                # f"Total Accuracy: {fullRight/length*100:.2f}%")
        torch.save({k: v for k, v in self.state_dict().items() if "mel_transform" not in k}, "transformer_weights.pth")