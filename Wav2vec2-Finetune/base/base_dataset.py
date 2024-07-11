import os
import string
import pandas as pd
import sys
import re
import librosa
import numpy as np
from pandarallel import pandarallel
from typing import Dict, List

# For testing 
sys.path.append('..')

from sklearn.model_selection import train_test_split
from utils.feature import load_wav
from tqdm import tqdm
from torch.utils.data import Dataset
from dataloader.dataset import Dataset as InstanceDataset


class BaseDataset(Dataset):
    def __init__(self, rank, dist, path, sr, delimiter, special_tokens, min_duration = -np.inf, max_duration = np.inf, preload_data = False, transform = None, nb_workers = 4, volume = None, model_type= "pinyin"):
        self.rank = rank
        self.dist = dist
        self.sr = sr
        self.volume = volume
        self.model_type = model_type
        # Special characters to remove in your data 
        self.chars_to_ignore = u"[！？。，＂＃％＆＇（）－／：；＜＝＞＠＼＾｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏"  + string.punctuation+']+'
        self.label  = ["[+]", "[++]", "[*]", "[SONANT]", "[MUSIC]", "[LAUGHTER]", "[ENS]", "[SYSTEM]"]
        self.transform = transform
        self.preload_data = preload_data
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.df = self.load_data(path, delimiter)
        self.special_tokens = special_tokens

        pandarallel.initialize(progress_bar=True, nb_workers = nb_workers)
        if min_duration != -np.inf or max_duration != np.inf:
            if self.rank == 0 and 'duration' not in self.df.columns:
                print("\n*****Generate duration column*****")
                self.df['duration'] = self.df['path'].parallel_apply(lambda filename: librosa.get_duration(filename=filename))
                #self.df.to_csv(path, index = False, sep = delimiter)
            self.dist.barrier()
            #self.df = self.load_data(path, delimiter)
            if self.rank == 0:
                print("\n*****Filter out invalid audio*****")
            mask = (self.df['duration'] <= self.max_duration) & (self.df['duration'] >= self.min_duration)
            self.df = self.df[mask]
        self.df['transcript'] = self.df['transcript'].parallel_apply(self.remove_special_characters)
        self.df = self.df[self.df['transcript'] != ""]
        self.df = self.df.reset_index(drop=True)
    
        if self.preload_data:
            if self.rank == 0:
                print(f"\n*****Preloading {len(self.df)} data*****")
            self.df['wav'] = self.df['path'].parallel_apply(lambda filepath: load_wav(filepath, sr = self.sr))
        
    def remove_special_characters(self, transcript) -> str:
        rule = re.compile(self.chars_to_ignore)
        label_pattern = '|'.join(map(re.escape, self.label))
        rule_label = re.compile(f'(\[{label_pattern}\])')

        transcript = re.sub(rule_label, "", transcript)    

        transcript = re.sub(rule, " ", transcript).lower()
        
        transcript = ' '.join(transcript.split())
        return transcript

    def get_vocab_dict(self) -> Dict[int, str]:
        # Read https://huggingface.co/blog/fine-tune-wav2vec2-english for more information
        all_text = " ".join(list(self.df["transcript"]))
        #  remove special tokens in all_text, otherwise it will tokenize the special tokens' characters. Eg: <unk> -> '<', 'u', 'n', 'k', '>'
        for v in self.special_tokens.values():
            all_text = all_text.replace(v, '')
        vocab_list = list(set(all_text))
        vocab_list.sort()
        vocab_dict = {v: k for k, v in enumerate(vocab_list)}

        vocab_dict["|"] = vocab_dict[" "]
        del vocab_dict[" "]
        for v in self.special_tokens.values():
            vocab_dict[v] = len(vocab_dict)
        #print(vocab_dict)
        return vocab_dict

    def preload_dataset(self, paths, sr) -> List:
        wavs = []
        print("Preloading {} data".format(self.mode))
        for path in tqdm(paths, total = len(paths)):
            wav = load_wav(path, sr)
            wavs += [wav]
        return wavs

    def load_data(self,input_folders, delimiter) -> pd.DataFrame:
        all_paths = []
        all_transcripts = []
        input_folders = input_folders.split(',')
        for input_folder in input_folders :
            if os.path.isdir(input_folder):
                all_files = os.listdir(input_folder)

                wav_files = [f for f in all_files if f.endswith(".wav")]

                for wav_file in wav_files:
                    
                    wav_path = os.path.join(input_folder, wav_file)
                    if self.model_type == "pinyin" :
                        print("training on pinyin")
                        transcript_path = os.path.join(input_folder, os.path.splitext(wav_file)[0] + "_pipyin.txt")
                    else :
                        print("training on hanzi")
                        transcript_path = os.path.join(input_folder, os.path.splitext(wav_file)[0] + ".txt")
                    if os.path.exists(transcript_path):
                        with open(transcript_path, 'r', encoding='utf-8') as transcript_file:
                            transcript = transcript_file.read()

                        all_paths.append(wav_path)
                        all_transcripts.append(transcript)

            subfolders = [f for f in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, f))]

            for subfolder in subfolders:
                subfolder_path = os.path.join(input_folder, subfolder)

                all_files = os.listdir(subfolder_path)

                wav_files = [f for f in all_files if f.endswith(".wav")]

                for wav_file in wav_files:
                    wav_path = os.path.join(subfolder_path, wav_file)
                    if self.model_type == "pinyin" :
                        transcript_path = os.path.join(subfolder_path, os.path.splitext(wav_file)[0] + "_pipyin.txt")
                    else : 
                        transcript_path = os.path.join(subfolder_path, os.path.splitext(wav_file)[0] + ".txt")
                    if os.path.exists(transcript_path):
                        with open(transcript_path, 'r', encoding='utf-8') as transcript_file:
                            transcript = transcript_file.read()

                        all_paths.append(wav_path)
                        all_transcripts.append(transcript)
        
        df = pd.DataFrame({'path': all_paths, 'transcript': all_transcripts})
        df = df.sample(frac=self.volume, random_state=42)
        return df

    def get_data(self) -> Dataset:
        ds = InstanceDataset(self.df, self.sr, self.preload_data, self.transform)
        return ds


if __name__ == '__main__':
    ds = BaseDataset(
        path = '/home/pvanh/data/zh_stt/ASR-RAMC-BIGCCSC', 
        sr = 16000, 
        preload_data = False,
        rank = 1,
        dist=None,
        delimiter="|",
        special_tokens=None, 
        # val_size = None, 
        transform = None)
    df= ds.load_data()
    print(df.head)
    vocab_dict = ds.get_vocab_dict()
    for k, v in vocab_dict.items():
        print(f'{k} - {v}')
