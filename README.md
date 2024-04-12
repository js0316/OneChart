<h3><a href="">OneChart: Purify the Chart Structural Extraction via One Auxiliary Token</a></h3>
<a href=""><img src="https://img.shields.io/badge/Paper-PDF-orange"></a> 
<a href=""><img src="https://img.shields.io/badge/demo-blue"></a> 
<a href=""><img src="https://img.shields.io/badge/zhihu-yellow"></a> 

Jinyue Chen*, Lingyu Kong*, [Haoran Wei](https://scholar.google.com/citations?user=J4naK0MAAAAJ&hl=en), Chenglong Liu, [Zheng Ge](https://joker316701882.github.io/), Liang Zhao, [Jianjian Sun](https://scholar.google.com/citations?user=MVZrGkYAAAAJ&hl=en), Chunrui Han, [Xiangyu Zhang](https://scholar.google.com/citations?user=yuB-cfoAAAAJ&hl=en)
	


<p align="center">
<img src="assets/logo.png" style="width: 100px" align=center>
</p>

## Release
- [2024/4/11] We have released the codes, weights and the ChartSE benchmark. 

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/DATA_LICENSE)
**Usage and License Notices**: The data, code, and checkpoint are intended and licensed for research use only. They are also restricted to use that follow the license agreement of Vary, LLaMA, Vicuna, Qwen, and LLaVA. 

## Benchmark Data and Evaluation Tool
1. Download the ChartSE images and jsons [here](). 
2. Modify json path at the begining of `ChartSE_eval/eval_ChartSE.py`. Then run eval script:
   
```shell
python ChartSE_eval/eval_ChartSE.py
```

## Install
1. Clone this repository and navigate to the code folder
```bash
git clone https://github.com/LingyvKong/OneChart.git
cd OneChart/OneChart_code/
```
2. Install Package
```Shell
conda create -n onechart python=3.10 -y
conda activate vary
pip install -e .
pip install -r requirements.txt
pip install ninja
pip install flash-attn --no-build-isolation
```


## Weights
Download the OneChart weights [here](https://huggingface.co/kppkkp/OneChart/tree/main). 
  
## Demo
```Shell
python vary/demo/run_opt_v1.py  --model-name  /onechart_weights_path/
```
Following the instruction, type `1` first, then type image path.

## Train
Prepare the dataset and fill in the data path to `OneChart/OneChart_code/vary/utils/constants.py`. Then a example script is:
```shell
deepspeed /data/OneChart_code/vary/train/train_opt.py     --deepspeed /data/OneChart_code/zero_config/zero2.json --model_name_or_path /data/checkpoints/varytiny/  --vision_tower /data/checkpoints/varytiny/ --freeze_vision_tower False --freeze_lm_model False --vision_select_layer -2 --use_im_start_end True --bf16 True --per_device_eval_batch_size 4 --gradient_accumulation_steps 1 --evaluation_strategy "no" --save_strategy "steps" --save_steps 250 --save_total_limit 1 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type "cosine" --logging_steps 1 --tf32 True --model_max_length 2048 --gradient_checkpointing True --dataloader_num_workers 4 --report_to none --per_device_train_batch_size 16 --num_train_epochs 1 --learning_rate 5e-5 --datasets render_chart_en+render_chart_zh  --output_dir /data/checkpoints/onechart-pretrain/
```
You can pay attention to modifying these parameters according to your needs: `--model_name_or_path`, `freeze_vision_tower`, `--datasets`, `--output_dir`


## Acknowledgement
- [Vary](https://github.com/Ucas-HaoranWei/Vary): the codebase and initial weights we built upon!




## Citation
If you find our work useful in your research, please consider citing OneChart:
```bibtex
@article{chen2024onechart,
  title={OneChart: Purify the Chart Structural Extraction via One Auxiliary Token},
  author={Chen, Jinyue and Kong, Lingyu and Wei, Haoran and Liu, Chenglong and Ge, Zheng and Zhao, Liang and Sun, Jianjian and Han, Chunrui and Zhang, Xiangyu},
  journal={arXiv preprint arXiv:},
  year={2024}
}
```