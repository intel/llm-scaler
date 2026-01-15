import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import AutoModel
import torchaudio

def cosyvoice3_example():
    # Using the mounted model path
    cosyvoice = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B')
    
    # zero_shot usage
    # Ensure ./asset/zero_shot_prompt.wav exists (it is part of the repo)
    for i, j in enumerate(cosyvoice.inference_zero_shot('八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮。', 'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。',
                                                        './asset/zero_shot_prompt.wav', stream=False)):
        torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
        print(f"Saved zero_shot_{i}.wav")

if __name__ == '__main__':
    cosyvoice3_example()
