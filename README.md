## ESWA_CMFAEN
The official implementation for "**A cross-modal feature aggregation and enhancement network for hyperspectral and LiDAR joint classification**", Expert Systems With Applications(ESWA), 2024.
![CMFAEN](https://github.com/zhangyiyan001/CMFAEN/blob/main/CMFAEN_framework.png)
****

## How to run it
```
config = {
    'dataset': 'Houston',  # Choose the dataset. 'Houston', 'Trento', 'MUUFL'
    'channel': 144,        # Hosuton->144; Trento->63; MUUFL->64
    'class_num': 15,       # Hosuton->15;  Trento->6;  MUUFL->11
    'batch_size': 128,
    'window_size': 11,
    'learning_rate': 0.0005,
    'weight_decay': 0.0001,
    'epoches': 300,
    'lr_step_size': 30,
    'lr_gamma': 0.5,
    'device': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    'seed': 100
}
```
```
python main.py
```
****
## Citation
If you find our work helpful in your research, kindly consider citing it. We appreciate your support！
```
@article{zhang2024cross,
  title={A cross-modal feature aggregation and enhancement network for hyperspectral and LiDAR joint classification},
  author={Zhang, Yiyan and Gao, Hongmin and Zhou, Jun and Zhang, Chenkai and Ghamisi, Pedram and Xu, Shufang and Li, Chenming and Zhang, Bing},
  journal={Expert Systems with Applications},
  pages={125145},
  year={2024},
  publisher={Elsevier}
}
```
## Contact
If you have any questions or suggestions, feel free to contact me.  
Email: zhangyiyan@hhu.edu.cn 

## License  
This project is released under the [Apache 2.0 license](LICENSE).
