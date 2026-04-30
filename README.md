Automated segmentation of fungal structures in fluorescence microscopy images plays a vital role 
in plant pathology research, particularly for understanding vine wood diseases such as Esca. This 
paper presents a systematic comparative study of deep learning-based image segmentation 
models applied to the vinewood fluorescence microscopy dataset. Six model architectures — U
Net, ResU-Net, DenseU-Net, M-Net, MNet-MRF, and Swin Transformer — were evaluated both 
before and after hyperparameter tuning, yielding 14 experimental configurations in total. 
A unified hyperparameter tuning framework based on random search was implemented, 
evaluating 20 random configurations per model over key parameters including learning rate, batch 
size, optimizer, and dropout. Models were trained for up to 30 epochs with early stopping 
(patience = 5) to prevent overfitting. The best configurations were selected based on minimum 
validation loss. 
The experimental results demonstrate that after hyperparameter tuning, the U-Net achieved a test 
accuracy of 0.9530 and a Dice coefficient of 0.9534, ranking as the top-performing model in terms 
of balanced metrics. The Swin Transformer achieved the highest Val Dice of 0.9666 post-tuning. 
Overall, hyperparameter optimization improved model stability and reduced overfitting across all 
architectures. This work provides a reusable and extensible framework for deep learning 
segmentation research in microscopy imaging. 
Keywords: Image segmentation, fungal detection, U-Net, Swin Transformer, hyperparameter 
tuning, vinewood microscopy, deep learning, fluorescence microscopy
