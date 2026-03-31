#!/usr/bin/env python3
"""
Simplified inference script for ToothSeg model.

This script provides an easy-to-use interface for running tooth segmentation
on CBCT scans using pre-trained ToothSeg models. It handles all the preprocessing,
dual-branch prediction, and post-processing steps automatically.

Usage:
    python simple_inference.py --input_dir /path/to/data --output_dir /path/to/output

Requirements:
    - Pre-trained model checkpoints downloaded from Zenodo
    - Input data in nnU-Net format (NIfTI files ending with _0000.nii.gz)
    - Environment variables set: nnUNet_results, nnUNet_raw, nnUNet_preprocessed
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import shutil
from typing import Optional


class ToothSegInference:
    """Easy-to-use inference pipeline for ToothSeg."""

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        semantic_dataset_id: int = 121,
        instance_dataset_id: int = 123,
        num_processes: int = 8,
        gpu_id: Optional[int] = None,
        use_pretrained_121_123: bool = True,
    ):
        """
        Initialize ToothSeg inference pipeline.

        Args:
            input_dir: Directory containing imagesTs folder with test images
            output_dir: Directory to save predictions
            semantic_dataset_id: Dataset ID for semantic branch (default: 121 for ToothFairy2)
            instance_dataset_id: Dataset ID for instance branch (default: 123 for ToothFairy2)
            num_processes: Number of parallel processes for post-processing
            gpu_id: GPU device ID to use (None for all available GPUs)
            use_pretrained_121_123: Whether using pre-trained models from Zenodo (121/123)
        """
        self.input_dir = Path(input_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.semantic_dataset_id = semantic_dataset_id
        self.instance_dataset_id = instance_dataset_id
        self.num_processes = num_processes
        self.gpu_id = gpu_id
        self.use_pretrained = use_pretrained_121_123

        # Check environment variables
        self._check_environment()

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Define subdirectories
        self.resized_dir = self.input_dir / "imagesTs_resized_for_instanceseg_spacing_02_02_02"
        self.semseg_dir = self.output_dir / "semseg_branch"
        self.instseg_bordercore_dir = self.output_dir / "instseg_branch_border_core"
        self.instseg_instances_dir = self.output_dir / "instseg_branch_instances"
        self.instseg_resized_dir = self.output_dir / "instseg_branch_instances_resized"
        self.final_dir = self.output_dir / "final_prediction"

    def _check_environment(self):
        """Check required environment variables and dependencies."""
        required_vars = ['nnUNet_results', 'nnUNet_raw', 'nnUNet_preprocessed']
        missing_vars = [var for var in required_vars if not os.getenv(var)]

        if missing_vars:
            raise EnvironmentError(
                f"Missing required environment variables: {', '.join(missing_vars)}\n"
                "Please set nnU-Net environment variables. See: "
                "https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/set_environment_variables.md"
            )

        # Check if input directory exists
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")

        # Check if imagesTs exists
        images_ts = self.input_dir / "imagesTs"
        if not images_ts.exists():
            raise FileNotFoundError(
                f"imagesTs folder not found in {self.input_dir}\n"
                "Please organize your data with test images in an 'imagesTs' subfolder"
            )

        # Check if there are any images
        nifti_files = list(images_ts.glob("*_0000.nii.gz"))
        if not nifti_files:
            raise FileNotFoundError(
                f"No NIfTI files found in {images_ts}\n"
                "Files must be named like: case_001_0000.nii.gz"
            )

        print(f"Found {len(nifti_files)} test images")

    def _run_command(self, cmd: list, description: str):
        """Run a command and handle errors."""
        print(f"\n{'='*60}")
        print(f"{description}")
        print(f"{'='*60}")
        print(f"Running: {' '.join(cmd)}")

        # Set GPU environment variable if specified
        env = os.environ.copy()
        if self.gpu_id is not None:
            env['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)

        # Disable nnUNet compilation for faster startup
        env['nnUNet_compile'] = 'F'

        try:
            result = subprocess.run(
                cmd,
                check=True,
                env=env,
                capture_output=False,
                text=True
            )
            print(f"✓ {description} completed successfully")
            return result
        except subprocess.CalledProcessError as e:
            print(f"✗ {description} failed with error code {e.returncode}")
            raise

    def step1_resize_test_set(self):
        """Step 1: Resize test set to 0.2x0.2x0.2 spacing for instance branch."""
        if self.resized_dir.exists():
            print(f"Resized data already exists at {self.resized_dir}, skipping resize step")
            return

        # Find the resize script
        script_path = Path(__file__).parent.parent / "toothseg" / "test_set_prediction_and_eval" / "resize_test_set.py"

        if not script_path.exists():
            raise FileNotFoundError(f"Resize script not found at {script_path}")

        cmd = [
            sys.executable,
            str(script_path),
            "-i", str(self.input_dir / "imagesTs"),
            "-o", str(self.resized_dir)
        ]

        self._run_command(cmd, "Step 1: Resizing test set for instance branch")

    def step2_predict_semantic(self):
        """Step 2: Run semantic segmentation branch prediction."""
        if self.semseg_dir.exists() and list(self.semseg_dir.glob("*.nii.gz")):
            print(f"Semantic predictions already exist at {self.semseg_dir}, skipping")
            return

        self.semseg_dir.mkdir(parents=True, exist_ok=True)

        # Determine dataset and configuration based on pre-trained model
        if self.use_pretrained:
            # Using ToothFairy2 pre-trained models
            dataset_id = 121
            trainer = "nnUNetTrainer_onlyMirror01_DASegOrd0"
            config = "3d_fullres_resample_torch_256_bs8"
            fold = "5"
        else:
            # Using custom trained models
            dataset_id = self.semantic_dataset_id
            trainer = "nnUNetTrainer_onlyMirror01_DASegOrd0"
            config = "3d_fullres_resample_torch_256_bs8"
            fold = "all"

        cmd = [
            "nnUNetv2_predict",
            "--continue_prediction",
            "--save_probabilities",
            "-i", str(self.input_dir / "imagesTs"),
            "-o", str(self.semseg_dir),
            "-d", str(dataset_id),
            "-tr", trainer,
            "-c", config,
            "-f", fold,
            "-nps", "1"
        ]

        self._run_command(cmd, "Step 2: Running semantic branch prediction")

    def step3_predict_instance(self):
        """Step 3: Run instance segmentation branch prediction (border-core format)."""
        if self.instseg_bordercore_dir.exists() and list(self.instseg_bordercore_dir.glob("*.nii.gz")):
            print(f"Instance predictions already exist at {self.instseg_bordercore_dir}, skipping")
            return

        self.instseg_bordercore_dir.mkdir(parents=True, exist_ok=True)

        # Determine dataset and configuration based on pre-trained model
        if self.use_pretrained:
            # Using ToothFairy2 pre-trained models
            dataset_id = 123
            trainer = "nnUNetTrainer"
            config = "3d_fullres_resample_torch_192_bs8"
            fold = "5"
        else:
            # Using custom trained models
            dataset_id = self.instance_dataset_id
            trainer = "nnUNetTrainer"
            config = "3d_fullres_resample_torch_192_bs8"
            fold = "all"

        cmd = [
            "nnUNetv2_predict",
            "--continue_prediction",
            "-i", str(self.resized_dir),
            "-o", str(self.instseg_bordercore_dir),
            "-d", str(dataset_id),
            "-tr", trainer,
            "-c", config,
            "-f", fold
        ]

        self._run_command(cmd, "Step 3: Running instance branch prediction (border-core)")

    def step4_convert_bordercore_to_instances(self):
        """Step 4: Convert border-core predictions to instance segmentation."""
        if self.instseg_instances_dir.exists() and list(self.instseg_instances_dir.glob("*.nii.gz")):
            print(f"Converted instances already exist at {self.instseg_instances_dir}, skipping")
            return

        # Find the conversion script
        script_path = Path(__file__).parent.parent / "toothseg" / "postprocess_predictions" / "border_core_to_instances.py"

        if not script_path.exists():
            raise FileNotFoundError(f"Conversion script not found at {script_path}")

        cmd = [
            sys.executable,
            str(script_path),
            "-i", str(self.instseg_bordercore_dir),
            "-o", str(self.instseg_instances_dir),
            "-np", str(self.num_processes)
        ]

        self._run_command(cmd, "Step 4: Converting border-core to instances")

    def step5_resize_instances(self):
        """Step 5: Resize instance predictions back to original spacing."""
        if self.instseg_resized_dir.exists() and list(self.instseg_resized_dir.glob("*.nii.gz")):
            print(f"Resized instances already exist at {self.instseg_resized_dir}, skipping")
            return

        # Find the resize script
        script_path = Path(__file__).parent.parent / "toothseg" / "postprocess_predictions" / "resize_predictions.py"

        if not script_path.exists():
            raise FileNotFoundError(f"Resize script not found at {script_path}")

        cmd = [
            sys.executable,
            str(script_path),
            "-i", str(self.instseg_instances_dir),
            "-o", str(self.instseg_resized_dir),
            "-ref", str(self.input_dir / "imagesTs"),
            "-np", str(self.num_processes)
        ]

        self._run_command(cmd, "Step 5: Resizing instances to original spacing")

    def step6_assign_tooth_labels(self):
        """Step 6: Assign tooth labels using self-correction."""
        if self.final_dir.exists() and list(self.final_dir.glob("*.nii.gz")):
            print(f"Final predictions already exist at {self.final_dir}, skipping")
            return

        # Find the label assignment script
        script_path = Path(__file__).parent.parent / "toothseg" / "postprocess_predictions" / "assign_mincost_tooth_labels.py"

        if not script_path.exists():
            raise FileNotFoundError(f"Label assignment script not found at {script_path}")

        cmd = [
            sys.executable,
            str(script_path),
            "-ifolder", str(self.instseg_resized_dir),
            "-sfolder", str(self.semseg_dir),
            "-o", str(self.final_dir),
            "-np", str(self.num_processes)
        ]

        self._run_command(cmd, "Step 6: Assigning tooth labels (self-correction)")

    def run(self):
        """Run the complete inference pipeline."""
        print("\n" + "="*60)
        print("ToothSeg Inference Pipeline")
        print("="*60)
        print(f"Input directory:  {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"GPU: {self.gpu_id if self.gpu_id is not None else 'All available'}")
        print(f"Parallel processes: {self.num_processes}")
        print("="*60 + "\n")

        try:
            # Run all steps
            self.step1_resize_test_set()
            self.step2_predict_semantic()
            self.step3_predict_instance()
            self.step4_convert_bordercore_to_instances()
            self.step5_resize_instances()
            self.step6_assign_tooth_labels()

            print("\n" + "="*60)
            print("✓ Inference completed successfully!")
            print("="*60)
            print(f"\nFinal predictions saved to: {self.final_dir}")
            print("\nEach segmentation mask uses FDI tooth numbering:")
            print("  11-18: Upper right quadrant")
            print("  21-28: Upper left quadrant")
            print("  31-38: Lower left quadrant")
            print("  41-48: Lower right quadrant")
            print("\n" + "="*60 + "\n")

        except Exception as e:
            print("\n" + "="*60)
            print("✗ Inference failed!")
            print("="*60)
            print(f"\nError: {str(e)}")
            print("\nPlease check:")
            print("  1. Environment variables are set correctly")
            print("  2. Pre-trained models are downloaded and placed in nnUNet_results")
            print("  3. Input data is in correct format (NIfTI files with _0000.nii.gz suffix)")
            print("  4. Sufficient disk space and GPU memory available")
            print("\n" + "="*60 + "\n")
            raise


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="ToothSeg: Easy-to-use inference for tooth segmentation in CBCT scans",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with pre-trained models
  python simple_inference.py --input_dir /path/to/data --output_dir /path/to/output

  # Specify GPU
  python simple_inference.py --input_dir /path/to/data --output_dir /path/to/output --gpu_id 0

  # Use custom trained models
  python simple_inference.py --input_dir /path/to/data --output_dir /path/to/output \\
      --semantic_dataset 181 --instance_dataset 188 --no_pretrained

  # Increase parallelization
  python simple_inference.py --input_dir /path/to/data --output_dir /path/to/output --num_processes 16

Input data structure:
  input_dir/
  └── imagesTs/
      ├── case_001_0000.nii.gz
      ├── case_002_0000.nii.gz
      └── ...

Output structure:
  output_dir/
  ├── semseg_branch/              # Semantic segmentation predictions
  ├── instseg_branch_*/           # Instance segmentation (intermediate)
  └── final_prediction/           # Final predictions (USE THIS!)
      ├── case_001.nii.gz
      ├── case_002.nii.gz
      └── ...
        """
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory containing imagesTs folder with test images in NIfTI format"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for predictions"
    )

    parser.add_argument(
        "--semantic_dataset",
        type=int,
        default=121,
        help="Dataset ID for semantic branch (default: 121 for pre-trained ToothFairy2 model)"
    )

    parser.add_argument(
        "--instance_dataset",
        type=int,
        default=123,
        help="Dataset ID for instance branch (default: 123 for pre-trained ToothFairy2 model)"
    )

    parser.add_argument(
        "--num_processes",
        type=int,
        default=8,
        help="Number of parallel processes for post-processing (default: 8)"
    )

    parser.add_argument(
        "--gpu_id",
        type=int,
        default=None,
        help="GPU device ID to use (default: use all available GPUs)"
    )

    parser.add_argument(
        "--no_pretrained",
        action="store_true",
        help="Don't use pre-trained models (use custom trained models instead)"
    )

    args = parser.parse_args()

    # Create and run inference pipeline
    pipeline = ToothSegInference(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        semantic_dataset_id=args.semantic_dataset,
        instance_dataset_id=args.instance_dataset,
        num_processes=args.num_processes,
        gpu_id=args.gpu_id,
        use_pretrained_121_123=not args.no_pretrained
    )

    pipeline.run()


if __name__ == "__main__":
    main()
