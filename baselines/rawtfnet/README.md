# RawTFNet-Pytorch
Official Pytorch Implementation of RawTFNet, A Lightweight CNN Architecture for Speech Anti-spoofing.
Please refer  to our paper [![arXiv](https://img.shields.io/badge/arXiv-2504.05657-b31b1b.svg)](https://arxiv.org/pdf/2507.08227)




## Prepare:

  1. Git clone this repo.
  2. Build environment:
     ```
     conda env create -f environment.yml
     ```
     or
     ```
     pip install -r requirements.txt
     ```
     * You may need to adjust the library versions according to your CUDA version and GPU spec.

## Dataset:
     
   If you want to train the model: The ASVspoof 2019 dataset can be downloaded from [here](https://datashare.ed.ac.uk/handle/10283/3336).
     
   If you want to test on the ASVspoof 2021 database, it is released on the zenodo site.
     
   -- LA [here](https://zenodo.org/records/4837263#.YnDIinYzZhE)
     
   -- DF [here](https://zenodo.org/records/4835108#.YnDIb3YzZhE)
     
   -- keys (labels) and metadata [here](https://www.asvspoof.org/index2021.html)
   
   If you want to test on the In-the-Wild dataset, it can be downloaded from [here](https://deepfake-total.com/in_the_wild)

## Usage:
     
### If you want to train the model by yourself on ASVspoof19 dataset:
     
  check the command template: 
  ```
  CUDA_VISIBLE_DEVICES=0 python main.py --track=LA --lr=0.0001 --batch_size=32 \
  --algo 4 --date 0520 --seed 12345 --loss=WCE --model_name rawtfnet \
  --num_epochs 100 --pool_func 'mean'\
  --database_path /your_path
  ```
  * Change the ```--database_path``` to your ASVspoof dataset path. 

     
### If you want to test on the ASVspoof 21 LA or DF dataset using the released pre-trained models or your own trained model:
     
  check the command template in: 
  ```
  test.sh
  ```
  1. For single model or averaged pretrained model:
     For the ASVspoof 21 LA, an example:
     ```
     python main.py --track=LA --batch_size=2 --is_eval --eval --model_name rawtfnet --pool_func 'mean' --test_protocol '4sec' \
     --database_path '/your_path' \
     --model_path="./ckpts/Best_RawTFNet_32.pth" \
     --eval_output='score_LA.txt'
     ```
     For the ASVspoof 21 DF, change the above command with:
     ```
     --track=DF
     ```
     For already averaged model, simply change the model path:
     ```
     --model_path="./ckpts/Best_RawTFNet_32.pth" \
     ```
     * Change the ```--database_path``` to your ASVspoof dataset path. 
     * Change the ```--model_path``` to your path of the checkpoint to test. You may use the checkpoint with the smallest validation EER for testing.
     * if you wish to average checkpoints and then test, pls refer to point 3 below
    
     And then, to get the final result of EER and minDCF:
     ```
     txtpath=score_LA.txt
     python evaluate_2021_LA.py $txtpath /home/user/database/keys eval
     ```
     and
     ```
     txtpath=score_DF.txt
     python evaluate_2021_DF.py $txtpath /home/user/database/keys eval
     ``` 
     
 3. For averaging checkpoints and then test, an example:
    ```
    python main.py --track=DF --batch_size=2 --is_eval --eval --model_name rawtfnet --pool_func 'mean' --test_protocol '4sec' \
    --num_average_model 5 --model_ID_to_average 56 60 62 76 95 \
    --database_path '/database/LA/' \
    --model_folder_path="./models" \
    --eval_output='score_DF_avg_ckpt_ep56_60_62_76_95.txt'
    ```
     * If you use checkpoints average function, choose the serveral epochs with smallest validation EERs for testing according to validation EER, and set as ``` --model_ID_to_average```.
     * change the ```--model_folder_path``` to the path of the folder that saving all checkpoints.
     * If you are using the pretrained model, these configs are the same as default.
 
    And then, similar as that of point 2 above, following the examples to get the EER and minDCF from the score text file.
    

## Reference Repo
Thanks for following open-source projects:
1. wav2vec2 + AASIST & Rawboost: https://github.com/TakHemlata/SSL_Anti-spoofing Paper: [[model]](https://arxiv.org/abs/2202.12233), [[Rawboost]](https://arxiv.org/abs/2202.12233)
2. tfsepconv: https://github.com/yqcai888/DCASE2023
3. nestnet: https://github.com/Liu-Tianchi/Nes2Net_ASVspoof_ITW

## Cite
```  
@article{xiao2025rawtfnet,
  title={RawTFNet: A Lightweight CNN Architecture for Speech Anti-spoofing},
  author={Xiao, Yang and Dang, Ting and Das, Rohan Kumar},
  journal={arXiv preprint arXiv:2507.08227},
  year={2025}
}
```
