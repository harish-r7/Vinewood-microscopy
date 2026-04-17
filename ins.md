# Activate virtual environment
.venv\Scripts\Activate

# Train
python run_model.py

# Tune 
python run_model_tuning.py swin
python run_model_tuning.py unet
python run_model_tuning.py mnet
python run_model_tuning.py resunet
python run_model_tuning.py denseunet
python run_model_tuning.py attentionunet
python run_model_tuning.py mnet_mrf
python run_model_tuning.py mnet_mrf_voting

