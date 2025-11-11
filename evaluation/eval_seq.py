# https://github.com/SadilKhan/Text2CAD/blob/main/Evaluation/eval_seq.py
import pandas as pd
import pickle
import os,sys
import argparse
sys.path.append("..")
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-2]))
from tqdm import tqdm
import traceback
import json

# Try importing CadSeqProc, provide fallbacks
try:
    from CadSeqProc.cad_sequence import CADSequence
    from CadSeqProc.utility.utils import (create_path_with_time, ensure_dir)
    from CadSeqProc.utility.logger import CLGLogger
    from rich import print
    CADSEQPROC_AVAILABLE = True
    csnLogger = CLGLogger().configure_logger().logger
except ImportError:
    print("Warning: CadSeqProc not available. Install from: https://github.com/SadilKhan/Text2CAD")
    CADSEQPROC_AVAILABLE = False

    # Fallback implementations for data loading and testing
    import logging
    import datetime

    csnLogger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    def create_path_with_time(base_path):
        """Fallback: create output path with timestamp"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{base_path}_{timestamp}"
        os.makedirs(output_path, exist_ok=True)
        return output_path

    def ensure_dir(path):
        """Fallback: ensure directory exists"""
        os.makedirs(path, exist_ok=True)

    class CADSequence:
        """Mock CADSequence for data format validation only"""
        @staticmethod
        def from_vec(vec, bit, denumericalize=False):
            raise NotImplementedError("CADSequence requires CadSeqProc installation")

        def generate_report(self, pred_cad, uid):
            raise NotImplementedError("generate_report requires CadSeqProc installation")


def main():
    parser=argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--input_path",help="Predicted CAD Sequence in pkl format",required=True)
    parser.add_argument("--output_dir",help="Output dir",required=True)
    parser.add_argument("--verbose",action='store_true')
    parser.add_argument("--validate_only",action='store_true',
                       help="Only validate data format without full evaluation")

    args=parser.parse_args()

    # Load data
    with open(args.input_path,"rb") as f:
        data=pickle.load(f)

    # If only validating data format
    if args.validate_only or not CADSEQPROC_AVAILABLE:
        if not CADSEQPROC_AVAILABLE:
            print("\nCadSeqProc not available - running data format validation only")
        validate_data_format(data, args.verbose)
        return

    # Full evaluation with CadSeqProc
    output_dir=create_path_with_time(args.output_dir)

    if args.verbose:
        csnLogger.info("Evaluation for Design History")
        csnLogger.info(f"Output Path {output_dir}")

    for level in range(1,5):
        csnLogger.info(f"Level {level}")
        output_path=os.path.join(output_dir,'level_'+str(level))
        ensure_dir(output_path)
        generate_analysis_report(data=data,output_path=output_path,
                                logger=csnLogger,verbose=args.verbose, level='level_'+str(level))


def validate_data_format(data, verbose=True):
    """Validate pickle data format without CadSeqProc"""
    print("=" * 60)
    print("Data Format Validation")
    print("=" * 60)

    uids = list(data.keys())
    print(f"\n✓ Found {len(uids)} UIDs")

    if verbose and len(uids) > 0:
        print(f"\n  Sample UIDs: {uids[:3]}")

    # Check structure
    errors = []
    for uid in uids:
        if not isinstance(data[uid], dict):
            errors.append(f"UID {uid}: not a dict")
            continue

        for level_key in data[uid].keys():
            level_data = data[uid][level_key]

            # Check required keys
            required_keys = ['gt_cad_vec', 'pred_cad_vec', 'cd']
            for key in required_keys:
                if key not in level_data:
                    errors.append(f"UID {uid}/{level_key}: missing '{key}'")

            # Validate shapes
            if 'gt_cad_vec' in level_data:
                gt_vec = level_data['gt_cad_vec']
                if not isinstance(gt_vec, (list, tuple)) and not hasattr(gt_vec, 'shape'):
                    errors.append(f"UID {uid}/{level_key}: gt_cad_vec invalid type")

            if 'pred_cad_vec' in level_data:
                pred_vecs = level_data['pred_cad_vec']
                if not isinstance(pred_vecs, list):
                    errors.append(f"UID {uid}/{level_key}: pred_cad_vec should be list")

    if errors:
        print(f"\n✗ Found {len(errors)} format errors:")
        for err in errors[:10]:
            print(f"  - {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
    else:
        print("\n✓ Data format validation passed!")

    # Print statistics
    if len(uids) > 0:
        sample_uid = uids[0]
        levels = list(data[sample_uid].keys())
        print(f"\n✓ Levels per UID: {levels}")

        if len(levels) > 0:
            sample_level = levels[0]
            sample_data = data[sample_uid][sample_level]

            if 'gt_cad_vec' in sample_data:
                import numpy as np
                gt = sample_data['gt_cad_vec']
                if hasattr(gt, 'shape'):
                    print(f"  GT CAD vec shape: {gt.shape}")
                    print(f"  GT CAD vec dtype: {gt.dtype if hasattr(gt, 'dtype') else type(gt)}")

            if 'pred_cad_vec' in sample_data:
                preds = sample_data['pred_cad_vec']
                print(f"  Number of predictions: {len(preds)}")
                if len(preds) > 0 and hasattr(preds[0], 'shape'):
                    print(f"  Pred CAD vec shape: {preds[0].shape}")

            if 'cd' in sample_data:
                cds = sample_data['cd']
                print(f"  Chamfer distances: {cds}")

    print("\n" + "=" * 60)
    print("Note: Full evaluation requires CadSeqProc")
    print("Install from: https://github.com/SadilKhan/Text2CAD")
    print("=" * 60)


def generate_analysis_report(data,output_path,logger,verbose,level):
    report_df = pd.DataFrame() # Dataframe for analysis
    # cm=np.zeros((4,4)) # Confusion Matrix

    uids=list(data.keys())

    for uid in tqdm(uids):
        best_report_df=process_uid_(uid,data,level=level)
        if best_report_df is not None:
            report_df=pd.concat([report_df,best_report_df])
    csv_path=os.path.join(output_path,f"report_df_{level}.csv")

    try:
        report_df.to_csv(csv_path, index=None)
        # logger.success(f"Report is saved at {csv_path}")
    except Exception as e:
        logger.error(f"Error saving csv file at {csv_path}")
        if verbose:
           print(traceback.print_exc())

    if verbose:
        logger.info("Calculating Metrics...")

    eval_dict = {}

    line_metrics = report_df[(report_df['line_total_gt'] > 0)][['line_recall', 'line_precision', 'line_f1']].mean() * 100
    eval_dict['line'] = {
        'recall': line_metrics['line_recall'],
        'precision': line_metrics['line_precision'],
        'f1': line_metrics['line_f1']
    }

    # Mean Recall, Precision, F1 for Arc
    arc_metrics = report_df[(report_df['arc_total_gt'] > 0)][['arc_recall', 'arc_precision', 'arc_f1']].mean() * 100
    eval_dict['arc'] = {
        'recall': arc_metrics['arc_recall'],
        'precision': arc_metrics['arc_precision'],
        'f1': arc_metrics['arc_f1']
    }

    # Mean Recall, Precision, F1 for Circle
    circle_metrics = report_df[(report_df['circle_total_gt'] > 0)][['circle_recall', 'circle_precision', 'circle_f1']].mean() * 100
    eval_dict['circle'] = {
        'recall': circle_metrics['circle_recall'],
        'precision': circle_metrics['circle_precision'],
        'f1': circle_metrics['circle_f1']
    }

    # Mean Recall, Precision, F1 for Extrusion
    ext_recall = report_df['num_ext'] / report_df['num_ext_gt']
    ext_precision = report_df['num_ext'] / report_df['num_ext_pred']
    ext_f1 = 2 * ext_recall * ext_precision / (ext_recall + ext_precision)
    extrusion_metrics = {
        'recall': ext_recall.mean() * 100,
        'precision': ext_precision.mean() * 100,
        'f1': ext_f1.mean() * 100
    }
    eval_dict.update({'extrusion': extrusion_metrics})

    
    # Update Chamfer Distance
    eval_dict['cd']={}
    eval_dict['cd']['median']=report_df['cd'][report_df['cd']>0].median()
    eval_dict['cd']['mean']=report_df['cd'][report_df['cd']>0].mean()
    eval_dict['invalidity_ratio_percentage']=report_df['cd'][report_df['cd']<0].count()*100/len(report_df)

    if verbose:
        json_formatted_str = json.dumps(eval_dict, indent=4)
        print(json_formatted_str)

    mean_report_path=os.path.join(output_path,f"mean_report_{level}.json")

    with open(mean_report_path,"w") as f:
        json.dump(eval_dict,f, indent=4)



def process_vec(pred_vec,gt_vec,bit,uid):
    try:
        pred_cad=CADSequence.from_vec(pred_vec,8,denumericalize=False)
        gt_cad=CADSequence.from_vec(gt_vec,8,denumericalize=False)

        report_df,cm=gt_cad.generate_report(pred_cad,uid)
        
        return report_df,cm
    except Exception as e:
        #print(e)
        return None,None

def process_uid_(uid,data,level):
    try:
        gt_vec = data[uid][level]['gt_cad_vec']
        all_cd = data[uid][level]['cd']
        best_index = 0
        pred_vec = data[uid][level]['pred_cad_vec'][best_index]
        df, _ = process_vec(pred_vec, gt_vec, 8, uid)
        df['cd'] = all_cd[best_index]

        return df

    except Exception as e:
        return None

if __name__=="__main__":
    main()