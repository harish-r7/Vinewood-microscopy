# run_model_tuning.py - Master script to run any model
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description='Run hyperparameter tuning for a specific model')
    parser.add_argument('model', type=str, choices=[
        'unet', 'mnet', 'swin', 'resunet', 'denseunet', 'attentionunet', 'mnet_mrf', 'mnet_mrf_voting'
    ], help='Model to tune')
    
    args = parser.parse_args()
    
    if args.model == 'unet':
        from tune_models.tune_unet import tune_unet
        tune_unet()
    elif args.model == 'mnet':
        from tune_models.tune_mnet import tune_mnet
        tune_mnet()
    elif args.model == 'swin':
        from tune_models.tune_swin import tune_swin
        tune_swin()
    elif args.model == 'resunet':
        from tune_models.tune_resunet import tune_resunet
        tune_resunet()
    elif args.model == 'denseunet':
        from tune_models.tune_denseunet import tune_denseunet
        tune_denseunet()
    elif args.model == 'attentionunet':
        from tune_models.tune_attentionunet import tune_attentionunet
        tune_attentionunet()
    elif args.model == 'mnet_mrf':
        from tune_models.tune_mnet_mrf import tune_mnet_mrf
        tune_mnet_mrf()
    elif args.model == 'mnet_mrf_voting':
        from tune_models.tune_mnet_mrf_voting import tune_mnet_mrf_voting
        tune_mnet_mrf_voting()

if __name__ == "__main__":
    main()