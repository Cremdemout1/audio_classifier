import transformer as t
import torch
import pandas as pd
import inputs as sound
from pathlib import Path
import record as rec

#   types of semantics:

#   1. request_command - Direct instructions or imperatives
#   2. personal_statement - Subjective thoughts, feelings, plans about oneself
#   3. factual_statement - Objective observations, states, facts
#   4. narrative - Descriptions of actions, events, people, temporal sequences

#   types of emotions

#   Anger - 1,463 (16.5%)
#   Calm - 192 (2.2%)
#   Disgust - 1,463 (16.5%)
#   Fear - 1,463 (16.5%)
#   Happy - 1,463 (16.5%)
#   Neutral - 1,183 (13.3%)
#   Sad - 1,463 (16.5%)
#   Surprised - 192 (2.2%)

def main():
    sound.create_audio_CSV("./labels.csv", "audio.csv")
    model: t.transformer = t.transformer(learning_rate=0.001, epochs=100)
    if Path("transformer_weights.pth").exists():
        model.load_state_dict(torch.load("transformer_weights.pth"), strict=False)
        model.eval()
    else:
        model.train()
        model.fit()
        model.eval()
    print("\nModel ready. You can now input sentences for evaluation.")

    while True:
        user_input = input("\nEnter input index (0-8881) or 'exit' to quit or 'rec' to record your own message: ")
        if user_input.lower() == "exit":
            break
        if (user_input.lower() == 'rec'):
            audio_tensor = rec.user_listen()
            # processed_tensor = model.preprocess_tensor(audio_tensor)
            # processed_tensor = processed_tensor.view(1, processed_tensor[-2], processed_tensor[-1])
            # model.inputs = processed_tensor  # has to be changed... make flag for when it is user inputted
            with torch.no_grad():
                semantic, emotion = model.predict(user_recording=audio_tensor, training=False)
            print(f"Predicted Semantic: {semantic}, Predicted Emotion: {emotion}")
        try:
            idx = int(user_input)
            if idx < 0 or idx >= len(model.inputs):
                print(f"Index out of range. Must be between 0 and {len(model.inputs)-1}.")
                continue
            df = pd.read_csv('labels.csv')
            trueSem = df.loc[idx, 'Semantic']
            trueEmo = df.loc[idx, 'Emotion']
            sentence = df.loc[idx, 'Text']
            semantic, emotion = model.predict(input=idx, training=False)
            print(sentence)
            print(f"Predicted Semantic: {semantic} | actual semantic: {trueSem}")
            print(f"Predicted Emotion: {emotion} | actual emotion: {trueEmo}")

        except ValueError:
            print("Invalid input. Please enter a number or 'exit'.")

main()