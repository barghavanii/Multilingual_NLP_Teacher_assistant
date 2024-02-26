import os

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig
from TTS.utils.manage import ModelManager
import wandb
import random

wandb.init(
    project="persian_tts",  # Updated project name
    name="Arman_21FEB",  # Run name, can be adjusted as needed
    sync_tensorboard=True,  # Enable TensorBoard syncing
    entity="bahareh-arghavani"  # Add your wandb username or team name here
)


# Logging parameters
RUN_NAME = "GPT_XTTS_v2.0_Arman_fa"
PROJECT_NAME = "XTTS_trainer"
DASHBOARD_LOGGER = "tensorboard"
LOGGER_URI = None
# start a new wandb run to track this script

# Set here the path that the checkpoints will be saved. Default: ./run/training/
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fa_run", "training")

# Training Parameters
OPTIMIZER_WD_ONLY_ON_WEIGHTS = True  # for multi-gpu training please make it False
START_WITH_EVAL = True  # if True it will star with evaluation
BATCH_SIZE = 3  # set here the batch size
GRAD_ACUMM_STEPS = 84  # set here the grad accumulation steps
# Note: we recommend that BATCH_SIZE * GRAD_ACUMM_STEPS need to be at least 252 for more efficient training. You can increase/decrease BATCH_SIZE but then set GRAD_ACUMM_STEPS accordingly.

# Define here the dataset that you want to use for the fine-tuning on.
config_dataset = BaseDatasetConfig(
    formatter="mozilla",
    dataset_name="Arman_male",
    path="/data/notebook_files/datasets/Armna_5s",
    meta_file_train="/data/notebook_files/datasets/Armna_5s/metadata.csv",
    language="fa",
)

# Add here the configs of the datasets
DATASETS_CONFIG_LIST = [config_dataset]

# Define the path where XTTS v2.0.1 files will be downloaded
CHECKPOINTS_OUT_PATH = os.path.join(OUT_PATH, "XTTS_v2.0_original_model_files/")
os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)


# DVAE files
DVAE_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
MEL_NORM_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"

# Set the path to the downloaded files
DVAE_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(DVAE_CHECKPOINT_LINK))
MEL_NORM_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(MEL_NORM_LINK))

# download DVAE files if needed
if not os.path.isfile(DVAE_CHECKPOINT) or not os.path.isfile(MEL_NORM_FILE):
    print(" > Downloading DVAE files!")
    ModelManager._download_model_files([MEL_NORM_LINK, DVAE_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True)


# Download XTTS v2.0 checkpoint if needed
TOKENIZER_FILE_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json"
XTTS_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth"

# XTTS transfer learning parameters: You we need to provide the paths of XTTS model checkpoint that you want to do the fine tuning.
TOKENIZER_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(TOKENIZER_FILE_LINK))  # vocab.json file
XTTS_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(XTTS_CHECKPOINT_LINK))  # model.pth file

# download XTTS v2.0 files if needed
if not os.path.isfile(TOKENIZER_FILE) or not os.path.isfile(XTTS_CHECKPOINT):
    print(" > Downloading XTTS v2.0 files!")
    ModelManager._download_model_files(
        [TOKENIZER_FILE_LINK, XTTS_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True
    )


# Training sentences generations
SPEAKER_REFERENCE = [
    "/data/notebook_files/datasets/test/wavs/Motaba_shakori_clip11.wav"  # speaker reference to be used in training test sentences
]
LANGUAGE = config_dataset.language


def main():
    # init args and config
    model_args = GPTArgs(
        max_conditioning_length=132300,  # 6 secs
        min_conditioning_length=66150,  # 3 secs
        debug_loading_failures=False,
        max_wav_length=255995,  # ~11.6 seconds
        max_text_length=200,
        mel_norm_file=MEL_NORM_FILE,
        dvae_checkpoint=DVAE_CHECKPOINT,
        xtts_checkpoint=XTTS_CHECKPOINT,  # checkpoint path of the model that you want to fine-tune
        tokenizer_file=TOKENIZER_FILE,
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )
    # define audio config
    audio_config = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)
    # training parameters config
    config = GPTTrainerConfig(
        output_path=OUT_PATH,
        model_args=model_args,
        run_name=RUN_NAME,
        project_name=PROJECT_NAME,
        run_description="""
            GPT XTTS training
            """,
        dashboard_logger=DASHBOARD_LOGGER,
        logger_uri=LOGGER_URI,
        audio=audio_config,
        batch_size=BATCH_SIZE,
        batch_group_size=48,
        eval_batch_size=BATCH_SIZE,
        num_loader_workers=8,
        eval_split_max_size=256,
        print_step=50,
        plot_step=100,
        log_model_step=1000,
        save_step=10000,
        save_n_checkpoints=1,
        save_checkpoints=True,
        # target_loss="loss",
        print_eval=False,
        # Optimizer values like tortoise, pytorch implementation with modifications to not apply WD to non-weight parameters.
        optimizer="AdamW",
        optimizer_wd_only_on_weights=OPTIMIZER_WD_ONLY_ON_WEIGHTS,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=5e-06,  # learning rate
        lr_scheduler="MultiStepLR",
        # it was adjusted accordly for the new step scheme
        lr_scheduler_params={"milestones": [50000 * 18, 150000 * 18, 300000 * 18], "gamma": 0.5, "last_epoch": -1},
        test_sentences=[
            {
                "text": "زین همرهان سست عناصر، دلم گرفت.",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
            {
                "text": "آنکه مدام به کار دیگران سرک می کشد و کنجکاو است تا ببیند آنها چه می کنند مانند سایه ایی بر دیوار است که مدام بدنبال ما می دود بدون آنکه از خود اختیاری داشته باشد",
                "speaker_wav": SPEAKER_REFERENCE,
                "language":LANGUAGE,
            },
            {
                "text": "بیا تا گل برافشانیم و می در ساغر اندازیم.",
                "speaker_wav":SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
            {
                "text": "راه آشتی را کسی باید بیابد که خود سبب جدایی شده است",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
            {
                "text": "بنی آدم اعضای یک پیکرند, که در آفرینش ز یک گوهرند.",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
            {
                "text": "سهام زندگی به 10 درصد و سهام بیتکوین گوگل به 33 درصد افزایش یافت.",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
            {
                "text": "یه دو دقه هم به حرفم گوش کن، نگو نگوشیدم و نحرفیدی.",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
            {
                "text": "مطالعات نشان داده‌اند که 40 درصد از دانش‌آموزان، در سال تحصیلی 2024-2023، در دوره‌های آنلاین شرکت کرده‌اند، که این امر نشان‌دهنده تغییر چشمگیری در روش‌های آموزشی در ایران می‌باشد.",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
            {
                "text": "من بودم و آبجی فوتینا، و حالا رپتی پتینا. این شعر یکی از اشعار معروف رو حوضی است که در کوچه بازار تهران زمزمه می شده است.",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
            {
                "text": "در بودجه سال 2025، دولت ایران اعلام کرد که 15 درصد از کل بودجه، معادل تقریباً 30 میلیارد دلار، به توسعه زیرساخت‌های حمل‌ونقل و 25 درصد دیگر، به بهبود بخش بهداشت و درمان اختصاص یافته است.",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
            {
                "text": "تحقیقات بازار نشان می‌دهد که فروش گوشی‌های هوشمند در ایران، در سه ماهه اول سال 2024، نسبت به مدت مشابه در سال قبل، 55 درصد افزایش یافته که این پدیده بیشتر به دلیل عرضه مدل‌های جدید و کاهش قیمت‌ها است.",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
            {
                "text": "سلام، خوشحالم که دوباره می‌بینمت؛ امیدوارم که این هفته برات خوب گذشته باشه.",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
            {
                "text": "برای جشن تولد پدرم قصد دارم یک مهمانی کوچک با خانواده و دوستان نزدیک ترتیب بدهم ؛ دوست داری تو هم بیای؟",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
            {
                "text": "دیشب یک فیلم جذاب در تلویزیون دیدم که در مورد تاریخچه تمدن‌های باستان بود؛ واقعاً اطلاعات مفید و جالبی داشت.",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },

        ],
    )
    # init the model from config
    model = GPTTrainer.init_from_config(config)

    # load training samples
    train_samples, eval_samples = load_tts_samples(
        DATASETS_CONFIG_LIST,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    # init the trainer and 🚀
    trainer = Trainer(
        TrainerArgs(
            restore_path=None,  # xtts checkpoint is restored via xtts_checkpoint key so no need of restore it using Trainer restore_path parameter
            skip_train_epoch=False,
            start_with_eval=START_WITH_EVAL,
            grad_accum_steps=GRAD_ACUMM_STEPS,
        ),
        config,
        output_path=OUT_PATH,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()
    # [optional] finish the wandb run, necessary in notebooks
    wandb.finish()


if __name__ == "__main__":
    main()
