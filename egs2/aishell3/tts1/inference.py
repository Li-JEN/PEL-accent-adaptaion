from espnet2.bin.tts_inference import Text2Speech
import glob
import kaldiio
import numpy as np
import torch
import soundfile as sf
import time
from opencc import OpenCC
import string
from pypinyin.style._utils import get_finals, get_initials
from pypinyin import pinyin
# without vocoder
# tts = Text2Speech.from_pretrained(model_file="/path/to/model.pth")
# wav = tts("Hello, world")["wav"]

# with local vocoder
# tts = Text2Speech.from_pretrained(model_file="/path/to/model.pth", vocoder_file="/path/to/vocoder.pkl")
# wav = tts("Hello, world")["wav"]

# with pretrained vocoder (use ljseepch style melgan as an example)
cc = OpenCC('t2s')
text2speech = Text2Speech.from_pretrained(
train_config="/home/espnet/egs2/aishell3/tts1/exp/tts_train_raw_phn_pypinyin_g2p_phone/config.yaml",
model_file="/home/espnet/egs2/aishell3/tts1/exp/tts_train_raw_phn_pypinyin_g2p_phone/train.loss.best.pth",
vocoder_tag="parallel_wavegan/ljspeech_hifigan.v1",
device="cuda",
 )
model_dir= "/home/espnet/egs2/aishell3/tts1/exp/tts_train_raw_phn_pypinyin_g2p_phone"
spembs = None

if text2speech.use_spembs:
    xvector_ark = [p for p in glob.glob(f"{model_dir}/../../dump/**/spk_xvector.ark", recursive=True) if "tr" in p][0]
    xvectors = {k: v for k, v in kaldiio.load_ark(xvector_ark)}
    spks = list(xvectors.keys())

    # randomly select speaker
    # random_spk_idx = np.random.randint(0, len(spks))
    random_spk_idx = 2
    spk = spks[random_spk_idx]
    spembs = xvectors[spk].copy()
    print(f"selected spk: {spk}")

# Speaker ID selection
sids = None
if text2speech.use_sids:
    spk2sid = glob.glob(f"{model_dir}/../../dump/**/spk2sid", recursive=True)[0]
    with open(spk2sid) as f:
        lines = [line.strip() for line in f.readlines()]
    sid2spk = {int(line.split()[1]): line.split()[0] for line in lines}
    
    # randomly select speaker
    sids = np.array(2)
    # sids = np.array(np.random.randint(1, len(sid2spk)))
    spk = sid2spk[int(sids)]
    print(f"selected spk: {spk}")

# Reference speech selection for GST
speech = None
if text2speech.use_speech:
    # you can change here to load your own reference speech
    # e.g.
    # import soundfile as sf
    # speech, fs = sf.read("/home/TTS_data/chatbot_check/audio_file/split/101301/003.wav")
    # speech = torch.from_numpy(speech).float()
    speech = torch.randn(50000,) * 0.01

def frontend(text) -> str:
    # punct = string.punctuation
    full_punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~' + '→↓△▿⋄•！？?〞＃＄％＆』。（）＊＋，－╱︰；＜＝＞＠〔╲〕 ＿ˋ｛∣｝∼、〃》「」『』【】﹝﹞【】〝〞–—『』「」…'
    # text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.translate(str.maketrans('', '', full_punctuation))
    text = cc.convert(text)
    # print(text)
    return text


def tts(text,id):
    text = frontend(text)
    with torch.no_grad():
        start = time.time()
        wav = text2speech(text,speech=speech, spembs=spembs)["wav"]
    duration = time.time() - start
    rtf = (duration) / (len(wav) / text2speech.fs)
    sf.write(f"synthesis_chatbot_text/{id}.wav", wav.cpu().numpy(), text2speech.fs, "PCM_16")
    # print(f"RTF = {rtf:5f}, Duration: {duration:5f}, wav_len: {len(wav)}")

from tqdm import tqdm
from pypinyin import Style, pinyin
from pypinyin.style._utils import get_finals, get_initials
def g2p_phone(text):
    phones = [
        p
        for phone in pinyin(text, style=Style.TONE3)
        for p in [
            get_initials(phone[0], strict=True),
            get_finals(phone[0][:-1], strict=True) + phone[0][-1]
            if phone[0][-1].isdigit()
            else get_finals(phone[0], strict=True)
            if phone[0][-1].isalnum()
            else phone[0],
        ]
        # Remove the case of individual tones as a phoneme
        if len(p) != 0 and not p.isdigit()
    ]
    return phones
if __name__ == '__main__':
    ids = []
    test_text = []
    wavscp = open("data/pair/wav.scp", "w", encoding="utf-8")
    utt2spk = open("data/pair/utt2spk", "w", encoding="utf-8")
    text = open("data/pair/text", "w", encoding="utf-8")
    
    with open("/home/espnet/egs2/chatbot/tts1/data/all_phn/text", 'r') as src:
        all = src.readlines()
        for each in all:
             ids.append(each.split(" ")[0])

    with open("/home/espnet/egs2/chatbot/tts1/data/all/text", 'r') as src_text:
        all = src_text.readlines()
        # print(len(all))
        for each in all:
            [id, text_] = each.split(" ")
            if id in ids:
                test_text.append([id+'_china', ])
                wavscp.write("{} {}\n".format(id+'_china',f"/home/espnet/egs2/aishell3/tts1/synthesis_chatbot_text/{id}_china.wav"))
                utt2spk.write("{} {}\n".format(id+'_china', spk))
                text.write("{} {}\n".format(id+'_china', " ".join(g2p_phone(text_.strip("\n")))))
    # print(test_text)
    wavscp.close()
    utt2spk.close()
    text.close()
    # for id, text in tqdm(test_text,total=len(test_text)):
    #     tts(text,id)

    # tts(text)

    
