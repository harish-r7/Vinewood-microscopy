# generate_reports.py - Generate overfitting reports from saved logs (No emojis)
import pandas as pd
import os
from datetime import datetime

# Configuration
RESULTS_PATH = "./results/before_tuning"
EPOCH_LOG_PATH = "./results/epoch_logs"

def generate_overfitting_report(model_name, df, test_metrics=None):
    """Generate overfitting report from existing data"""
    model_name_clean = model_name.replace('-', '_').replace(' ', '_')
    report_path = os.path.join(RESULTS_PATH, f"{model_name_clean}_overfitting_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write(f"OVERFITTING ANALYSIS REPORT: {model_name}\n")
        f.write("="*60 + "\n\n")
        
        # Best metrics
        best_val_dice = df['val_dice'].max()
        best_val_acc = df['val_accuracy'].max()
        best_epoch_dice = df['val_dice'].idxmax() + 1
        best_epoch_acc = df['val_accuracy'].idxmax() + 1
        
        f.write("BEST METRICS:\n")
        f.write(f"  Best Val Dice: {best_val_dice:.4f} (Epoch {best_epoch_dice})\n")
        f.write(f"  Best Val Acc:  {best_val_acc:.4f} (Epoch {best_epoch_acc})\n\n")
        
        # Training progression
        f.write("TRAINING PROGRESSION:\n")
        f.write(f"  Start - Train Loss: {df['train_loss'].iloc[0]:.4f} -> End: {df['train_loss'].iloc[-1]:.4f}\n")
        f.write(f"  Start - Val Loss:   {df['val_loss'].iloc[0]:.4f} -> End: {df['val_loss'].iloc[-1]:.4f}\n")
        f.write(f"  Start - Val Dice:   {df['val_dice'].iloc[0]:.4f} -> End: {df['val_dice'].iloc[-1]:.4f}\n")
        f.write(f"  Start - Val Acc:    {df['val_accuracy'].iloc[0]:.4f} -> End: {df['val_accuracy'].iloc[-1]:.4f}\n\n")
        
        # Overfitting analysis
        final_train_loss = df['train_loss'].iloc[-1]
        final_val_loss = df['val_loss'].iloc[-1]
        loss_gap = final_val_loss - final_train_loss
        
        f.write("OVERFITTING ANALYSIS:\n")
        if loss_gap > 0.1:
            f.write(f"  WARNING: Possible overfitting!\n")
            f.write(f"     Validation loss ({final_val_loss:.4f}) > Training loss ({final_train_loss:.4f})\n")
            f.write(f"     Gap: {loss_gap:.4f}\n")
            f.write(f"  Suggestion: Increase dropout, add more augmentation, or reduce model complexity\n")
        elif loss_gap < -0.1:
            f.write(f"  Underfitting detected\n")
            f.write(f"     Validation loss ({final_val_loss:.4f}) < Training loss ({final_train_loss:.4f})\n")
            f.write(f"     Gap: {abs(loss_gap):.4f}\n")
            f.write(f"  Suggestion: Increase model capacity or train longer\n")
        else:
            f.write(f"  Good balance!\n")
            f.write(f"     Loss gap: {loss_gap:.4f}\n")
        
        # Learning curve analysis
        f.write(f"\nLEARNING CURVE ANALYSIS:\n")
        if len(df) > 10:
            last_5_val_loss = df['val_loss'].iloc[-5:].mean()
            prev_5_val_loss = df['val_loss'].iloc[-10:-5].mean() if len(df) >= 10 else last_5_val_loss
            
            if last_5_val_loss > prev_5_val_loss:
                f.write(f"  WARNING: Validation loss is increasing in last 5 epochs\n")
                f.write(f"     Last 5 epochs avg: {last_5_val_loss:.4f}\n")
                f.write(f"     Previous 5 epochs avg: {prev_5_val_loss:.4f}\n")
                f.write(f"  This indicates overfitting is occurring\n")
            else:
                f.write(f"  Validation loss is stable/decreasing\n")
        
        # Training vs Validation accuracy gap
        final_train_acc = df['train_accuracy'].iloc[-1]
        final_val_acc = df['val_accuracy'].iloc[-1]
        acc_gap = final_train_acc - final_val_acc
        
        f.write(f"\nACCURACY ANALYSIS:\n")
        f.write(f"  Final Train Accuracy: {final_train_acc:.4f}\n")
        f.write(f"  Final Val Accuracy:   {final_val_acc:.4f}\n")
        f.write(f"  Accuracy Gap: {acc_gap:.4f}\n")
        
        if acc_gap > 0.1:
            f.write(f"  WARNING: Large accuracy gap - possible overfitting\n")
        elif acc_gap < -0.05:
            f.write(f"  Validation accuracy higher than training - unusual but good\n")
        else:
            f.write(f"  Good accuracy balance\n")
        
        # Test results if available
        if test_metrics:
            f.write(f"\nFINAL TEST RESULTS:\n")
            f.write(f"  Test Accuracy: {test_metrics.get('accuracy', 0):.4f}\n")
            f.write(f"  Test Dice: {test_metrics.get('dice', 0):.4f}\n")
            f.write(f"  Test IoU: {test_metrics.get('iou', 0):.4f}\n")
            f.write(f"  Test F1: {test_metrics.get('f1', 0):.4f}\n")
        
        # Early stopping info
        if len(df) < 50:
            f.write(f"\nEarly stopping triggered at epoch {len(df)}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"  Report saved: {report_path}")
    return report_path

def load_test_results(model_name):
    """Load final test results if available"""
    model_name_clean = model_name.replace('-', '_').replace(' ', '_')
    results_file = os.path.join(RESULTS_PATH, f"{model_name_clean}_results.csv")
    
    if os.path.exists(results_file):
        df = pd.read_csv(results_file)
        if len(df) > 0:
            return df.iloc[0].to_dict()
    return None

def analyze_all_models():
    """Generate reports for all models that have epoch logs"""
    print("="*60)
    print("GENERATING OVERFITTING REPORTS FOR ALL MODELS")
    print("="*60)
    
    reports_generated = []
    models_found = []
    
    # Check which models have epoch logs
    if os.path.exists(EPOCH_LOG_PATH):
        for file in os.listdir(EPOCH_LOG_PATH):
            if file.endswith('_epoch_log.csv'):
                # Convert filename back to model name
                model_name = file.replace('_epoch_log.csv', '').replace('_', '-')
                models_found.append(model_name)
    
    if not models_found:
        print("\nNo epoch logs found in:", EPOCH_LOG_PATH)
        print("Please make sure you've run at least one model first.")
        return
    
    print(f"\nFound epoch logs for {len(models_found)} models:")
    for model in models_found:
        print(f"  - {model}")
    
    # Generate reports for each model
    print("\n" + "="*60)
    print("GENERATING REPORTS")
    print("="*60)
    
    for model_name in models_found:
        print(f"\nProcessing {model_name}...")
        
        model_name_clean = model_name.replace('-', '_').replace(' ', '_')
        epoch_log_file = os.path.join(EPOCH_LOG_PATH, f"{model_name_clean}_epoch_log.csv")
        
        if os.path.exists(epoch_log_file):
            df = pd.read_csv(epoch_log_file)
            test_metrics = load_test_results(model_name)
            
            generate_overfitting_report(model_name, df, test_metrics)
            reports_generated.append(model_name)
        else:
            print(f"  No epoch log found for {model_name}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nReports generated for {len(reports_generated)} models:")
    for model in reports_generated:
        model_name_clean = model.replace('-', '_').replace(' ', '_')
        report_file = os.path.join(RESULTS_PATH, f"{model_name_clean}_overfitting_report.txt")
        print(f"  - {model}: {report_file}")
    
    # Create a comparison summary
    create_comparison_summary(models_found)

def create_comparison_summary(models_found):
    """Create a comparison table of all models"""
    comparison_data = []
    
    for model_name in models_found:
        model_name_clean = model_name.replace('-', '_').replace(' ', '_')
        epoch_log_file = os.path.join(EPOCH_LOG_PATH, f"{model_name_clean}_epoch_log.csv")
        results_file = os.path.join(RESULTS_PATH, f"{model_name_clean}_results.csv")
        
        if os.path.exists(epoch_log_file):
            df = pd.read_csv(epoch_log_file)
            
            # Get best metrics
            best_val_dice = df['val_dice'].max()
            best_val_acc = df['val_accuracy'].max()
            final_train_loss = df['train_loss'].iloc[-1]
            final_val_loss = df['val_loss'].iloc[-1]
            loss_gap = final_val_loss - final_train_loss
            
            # Determine overfitting status
            if loss_gap > 0.1:
                overfit_status = "OVERFITTING"
            elif loss_gap < -0.05:
                overfit_status = "Underfitting"
            else:
                overfit_status = "Balanced"
            
            # Get test metrics if available
            test_acc = None
            test_dice = None
            if os.path.exists(results_file):
                results_df = pd.read_csv(results_file)
                if len(results_df) > 0:
                    test_acc = results_df.iloc[0].get('test_accuracy', None)
                    test_dice = results_df.iloc[0].get('test_dice', None)
            
            comparison_data.append({
                'Model': model_name,
                'Best Val Dice': f"{best_val_dice:.4f}",
                'Best Val Acc': f"{best_val_acc:.4f}",
                'Final Val Loss': f"{final_val_loss:.4f}",
                'Loss Gap': f"{loss_gap:.4f}",
                'Status': overfit_status,
                'Test Acc': f"{test_acc:.4f}" if test_acc else "N/A",
                'Test Dice': f"{test_dice:.4f}" if test_dice else "N/A",
                'Epochs': len(df)
            })
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Best Val Dice', ascending=False)
        
        summary_file = os.path.join(RESULTS_PATH, "models_comparison_summary.csv")
        comparison_df.to_csv(summary_file, index=False)
        
        print("\n" + "="*60)
        print("MODELS COMPARISON SUMMARY")
        print("="*60)
        print(comparison_df.to_string(index=False))
        print(f"\nComparison saved to: {summary_file}")

if __name__ == "__main__":
    analyze_all_models()