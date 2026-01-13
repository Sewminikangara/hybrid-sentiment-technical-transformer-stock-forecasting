"""
Training Progress Monitor
Check the status of hybrid model training
"""

import os
import sys
import glob
import pandas as pd
from pathlib import Path
from datetime import datetime

def check_training_progress():
    """Monitor hybrid model training progress"""
    
    print("=" * 80)
    print("HYBRID MODEL TRAINING PROGRESS MONITOR")
    print("=" * 80)
    print(f"Checked at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Check if training is running
    print("[1/5] Checking if training process is running...")
    running = os.popen("ps aux | grep train_hybrid_models.py | grep -v grep").read()
    if running:
        print("  ✓ Training is RUNNING")
        print(f"  Process: {running.strip()}")
    else:
        print("  ✗ Training is NOT running")
    
    # Check log file
    print("\n[2/5] Checking training log...")
    log_file = 'hybrid_training_log.txt'
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        if lines:
            print(f"  ✓ Log file found: {len(lines)} lines")
            
            # Find last stock being trained
            for line in reversed(lines[-100:]):
                if 'TRAINING' in line and ']' in line:
                    print(f"  Last activity: {line.strip()}")
                    break
            
            # Count completed stocks
            completed = [l for l in lines if '✓' in l and 'complete - 3 models trained' in l]
            errors = [l for l in lines if '✗ Error training' in l]
            
            print(f"\n  Stocks completed: {len(completed)}/9")
            print(f"  Errors encountered: {len(errors)}")
            
            # Show last 5 lines
            print("\n  Last 5 log lines:")
            for line in lines[-5:]:
                print(f"    {line.rstrip()}")
        else:
            print("  ⚠ Log file is empty (training may just be starting)")
    else:
        print("  ✗ No log file found")
    
    # Check saved models
    print("\n[3/5] Checking saved models...")
    model_patterns = [
        'results/*_early_fusion.pt',
        'results/*_late_fusion.pt',
        'results/*_attention_fusion.pt'
    ]
    
    total_models = 0
    for pattern in model_patterns:
        models = glob.glob(pattern)
        total_models += len(models)
        fusion_type = pattern.split('_')[-2] + '_' + pattern.split('_')[-1].replace('.pt', '')
        print(f"  {fusion_type}: {len(models)}/9 stocks")
    
    print(f"\n  Total models saved: {total_models}/27")
    
    # Check results files
    print("\n[4/5] Checking results files...")
    results_files = glob.glob('results/hybrid_training_results_*.csv')
    
    if results_files:
        latest_results = sorted(results_files)[-1]
        print(f"  ✓ Found {len(results_files)} results file(s)")
        print(f"  Latest: {os.path.basename(latest_results)}")
        
        try:
            df = pd.read_csv(latest_results)
            print(f"\n  Results summary:")
            print(f"    Models trained: {len(df)}/27")
            
            if len(df) > 0:
                print(f"\n  Performance by model:")
                summary = df.groupby('Model')[['RMSE', 'MAE', 'MAPE']].mean()
                print(summary.to_string())
                
                print(f"\n  Stocks trained:")
                for stock in df['Stock'].unique():
                    stock_models = df[df['Stock'] == stock]['Model'].tolist()
                    print(f"    {stock}: {', '.join(stock_models)}")
        except Exception as e:
            print(f"  ⚠ Could not read results file: {e}")
    else:
        print("  ✗ No results files found yet")
    
    # Estimate time remaining
    print("\n[5/5] Time estimate...")
    if total_models > 0:
        completed_pct = (total_models / 27) * 100
        print(f"  Progress: {completed_pct:.1f}% complete ({total_models}/27 models)")
        
        if total_models < 27:
            remaining = 27 - total_models
            # Rough estimate: ~15-20 min per model on CPU
            est_minutes = remaining * 17.5
            est_hours = est_minutes / 60
            print(f"  Estimated time remaining: {est_hours:.1f} hours ({est_minutes:.0f} minutes)")
        else:
            print("  ✓ TRAINING COMPLETE!")
    else:
        print("  Training just started, check back in 15-20 minutes")
    
    # Overall status
    print("\n" + "=" * 80)
    if total_models == 27:
        print("STATUS: ✓ TRAINING COMPLETE!")
        print("\nNext steps:")
        print("  1. Review results in results/hybrid_training_results_*.csv")
        print("  2. Run: python generate_all_plots.py")
        print("  3. Compare hybrid vs technical-only models")
    elif total_models > 0:
        print(f"STATUS: ⏳ IN PROGRESS ({completed_pct:.1f}% complete)")
        print("\nCheck progress again with:")
        print("  python check_training_progress.py")
    else:
        print("STATUS: ⚠ STARTING or NOT STARTED")
        if not running:
            print("\nTo start training:")
            print("  .venv/bin/python train_hybrid_models.py 2>&1 | tee hybrid_training_log.txt &")
    print("=" * 80)

if __name__ == "__main__":
    try:
        check_training_progress()
    except KeyboardInterrupt:
        print("\n\nProgress check interrupted")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError checking progress: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
