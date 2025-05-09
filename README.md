## Learning to Identify Seen, Unseen and Unknown in the Open World: A Practical Setting for Zero-Shot Learning
Pytorch implementation for the paper [**Learning to Identify Seen, Unseen and Unknown in the Open World: A Practical Setting for Zero-Shot Learning** - **WACV 2025**](https://openaccess.thecvf.com/content/WACV2025/papers/Parameswaran_Learning_to_Identify_Seen_Unseen_and_Unknown_in_the_Open_WACV_2025_paper.pdf)

## Execution
- The Data is provided in the [here](https://www.dropbox.com/scl/fo/sfpgot2r600zu08e4tm92/AO4VVLORH7W9i1AmwJHlqNw?rlkey=35p6e4iswlqzdn1511mau4o1s&st=bb93et8l&dl=0)

- Place flo_1024_attr.zip file in attr_data folder
- Unzip FLO.zip and place the files in a Dataset folder
- Unzip Stage2_data.zip and place the files in the Stage2_data folder

- Provide the path to datasets in the config/*.yaml files

- Run the exec.sh file in src folder to get the results. The results are stored in results_csv folder.

## Notes
- Simillarly, the experiments can be repeated for other datasets.
- We use the 1024 dimension attribute for CUB and FLO dataset.
- The synthetic data is generated via [GSMFlow](https://github.com/uqzhichen/GSMFlow)
- The implementation details are in [supplementary material](https://openaccess.thecvf.com/content/WACV2025/supplemental/Parameswaran_Learning_to_Identify_WACV_2025_supplemental.pdf)

## Acknowledgement
The work is built upon [GSMFlow](https://github.com/uqzhichen/GSMFlow) and [A Boundary Based Out-of-Distribution Classifier for Generalized Zero-Shot Learning](https://github.com/Chenxingyu1990/A-Boundary-Based-Out-of-Distribution-Classifier-for-Generalized-Zero-Shot-Learning) 

## Citation

If you find our work is useful in your research, please consider citing:

```bibtex
@inproceedings{parameswaran2025learning,
  title={Learning to Identify Seen, Unseen and Unknown in the Open World: A Practical Setting for Zero-Shot Learning},
  author={Parameswaran, Sethupathy and Fang, Yuan and Gautam, Chandan and Ramasamy, Savitha and Li, Xiaoli},
  booktitle={2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  pages={6868--6878},
  year={2025},
  organization={IEEE}
}
```

## Contact

If you have any questions or concerns, please send email to [sethupathyp@iisc.ac.in](mailto:sethupathyp@iisc.ac.in)

