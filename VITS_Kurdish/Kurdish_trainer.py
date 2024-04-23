import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import wandb
from trainer import Trainer, TrainerArgs

from TTS.tts.configs.shared_configs import BaseDatasetConfig , CharactersConfig
from TTS.config.shared_configs import BaseAudioConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsAudioConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

 # Start a wandb run with `sync_tensorboard=True`
#wandb.init(project="persian-tts-vits-grapheme-azure-fa", group="GPU 6,7 accel mixed fp16 64x64", sync_tensorboard=True)

# output_path = os.path.dirname(os.path.abspath(__file__))
# output_path = output_path + '/notebook_files/runs'
#output_path = wandb.run.dir
output_path = "ZD_output"

print("output path is:")
print(output_path)

cache_path = "cache"
dataset_config = BaseDatasetConfig(
    formatter="mozilla", meta_file_train="metadata.csv", path="/home/bargh1/ZD_Final"
)

character_config=CharactersConfig(
  characters=' ي  ء ا ب ت ث ج ح خ د ذ ر ز ژ س ش  ع غ ف ق ل م ن ه و ۆ ی ڕ چ ڕ گ ک پ ە ڤ ھ ێ ك ڵ ئ',
#   characters="!¡'(),-.:;¿?ABCDEFGHIJKLMNOPRSTUVWXYZabcdefghijklmnopqrstuvwxyzáçèéêëìíîïñòóôöùúûü«°±µ»$%&‘’‚“`”„",
  punctuations='!(),-.:;? ̠،؛؟‌<>',
  phonemes='ˈˌːˑpbtdʈɖcɟkɡqɢʔɴŋɲɳnɱmʙrʀⱱɾɽɸβfvθðszʃʒʂʐçʝxɣχʁħʕhɦɬɮʋɹɻjɰlɭʎʟaegiouwyɪʊ̩æɑɔəɚɛɝɨ̃ʉʌʍ0123456789"#$%*+/=ABCDEFGHIJKLMNOPRSTUVWXYZ[]^_{}',
  pad="<PAD>",
  eos="<EOS>",
  bos="<BOS>",
  blank="<BLNK>",
  characters_class="TTS.tts.models.vits.VitsCharacters",
  )

audio_config = BaseAudioConfig(
     sample_rate=22050,
     do_trim_silence=True,
     min_level_db=-1,
    # do_sound_norm=True,
     signal_norm=True,
     clip_norm=True,
     symmetric_norm=True,
     max_norm = 0.9,
     resample=True,
     win_length=1024,
     hop_length=256,
     num_mels=80,
     mel_fmin=0,
     mel_fmax=None 
 )

vits_audio_config = VitsAudioConfig(
    sample_rate=22050,
#    do_sound_norm=True,
    win_length=1024,
    hop_length=256,
    num_mels=80,
    # do_trim_silence=True, #from hugging
    mel_fmin=0,
    mel_fmax=None
)
config = VitsConfig(
    audio=vits_audio_config, #from huggingface
    run_name="persian-tts-vits-grapheme-azure",
    batch_size=16,
    batch_group_size=16,
    eval_batch_size=4,
    num_loader_workers=4,
    num_eval_loader_workers=2,
    run_eval=True,
    run_eval_steps = 200,
    print_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    save_step=200,
    text_cleaner="basic_cleaners", #from MH
    use_phonemes=False,
    # phonemizer='persian_mh', #from TTS github
    # phoneme_language="fa",
    characters=character_config, #test without as well
    phoneme_cache_path=os.path.join(cache_path, "phoneme_cache_grapheme_azure"),
    compute_input_seq_cache=True,
    print_step=200,
    mixed_precision=False, #from TTS - True causes error "Expected reduction dim"
    test_sentences=[
        ["دەتوانی لەم بەرهەمە دەخوێنیت بەشێوەیەکی خوشەویست."],
        ["ئەو پاشانی کاردەکات بە دڵخوازی و دەچێت بەهەڵە دڵی دوایی."],
        ["سەرەتا دەبێت بە هەرێمی نەخشەی بەکاربێنیت."],
    ],
    output_path=output_path,
    datasets=[dataset_config]
)

# INITIALIZE THE AUDIO PROCESSOR
# Audio processor is used for feature extraction and audio I/O.
# It mainly serves to the dataloader and the training loggers.
ap = AudioProcessor.init_from_config(config)

# INITIALIZE THE TOKENIZER
# Tokenizer is used to convert text to sequences of token IDs.
# config is updated with the default characters if not defined in the config.
tokenizer, config = TTSTokenizer.init_from_config(config)

# LOAD DATA SAMPLES
# Each sample is a list of ```[text, audio_file_path, speaker_name]```
# You can define your custom sample loader returning the list of samples.
# Or define your custom formatter and pass it to the `load_tts_samples`.
# Check `TTS.tts.datasets.load_tts_samples` for more details.
train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

# init model
model = Vits(config, ap, tokenizer, speaker_manager=None)

# init the trainer and 🚀

trainer = Trainer(
    TrainerArgs(use_accelerate=True),
    config,
    output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)
trainer.fit()
