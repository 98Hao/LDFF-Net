## Train
```
python train.py
Modify the training dataset path in the first line of the config.yml file. The folder structure is as follows:

---your train dataset path
     ---gt # Only the GT images are required; the LR images will be generated based on them.
          ---000.png
          ---001.png
```

## Test
```
python test.py
Modify the path in line 20 to the low-resolution images that need to be tested
Modify the path in line 21 to specify the directory for saving the test results
```

## Evaluation
```
python evaluation.py
Modify the paths in lines 11 and 12 to your GT and SR directories
```

## Acknowledgements

This code is built on [MSLRSR](https://github.com/clelevo/MSLRSR), [HIIF](https://github.com/YuxuanJJ/HIIF) and [HS-FPN](https://github.com/ShiZican/HS-FPN). We thank the authors for sharing their codes.

---
