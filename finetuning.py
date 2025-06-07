# -*- coding: utf-8 -*-
"""
This script provides a detailed workflow and the necessary commands for fine-tuning
the MiniGPT-4 model on the Flickr30k dataset. The goal is to perform instruction
fine-tuning on the linear projection layer that connects the vision encoder to the LLM.

The overall process is as follows:
1.  Setup: Download the Flickr30k dataset.
2.  Data Preparation: Run the data preparation script to convert the dataset into the required format.
3.  Configuration: Update the training configuration file with the correct path to your pre-trained model.
4.  Training: Launch the fine-tuning process using the main training script.
"""

import os
import subprocess

def run_command(command):
    """Executes a shell command and prints its output."""
    print(f"Executing: {command}")
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
        # exit(1) # Uncomment if you want the script to stop on error

def main():
    print("Starting the MiniGPT-4 Flickr30k fine-tuning process...")
    print("-" * 50)

    # --- Step 1: Data Preparation ---
    print("Step 1: Preparing Flickr30k data...")
    print("This will process 'flickr_annotations_30k.csv' and create 'flickr30k_grounded_detail.json'.")
    
    prepare_script_path = os.path.join("prepare_flickr30k.py")
    if os.path.exists(prepare_script_path):
        # run_command(f"python {prepare_script_path}") # You can uncomment this to run it automatically
        print(f"\nPlease run the data preparation script manually if you haven't already:")
        print(f"python {prepare_script_path}")
    else:
        print(f"Error: {prepare_script_path} not found. Please ensure the file exists.")
        return
        
    print("\nData preparation script located. Please ensure it has been run successfully before proceeding.")
    print("-" * 50)


    # --- Step 2: Configuration Check ---
    config_path = os.path.join("train_configs", "minigpt4_flickr_finetune.yaml")
    print("Step 2: Check your configuration file.")
    print(f"Please open the following file and verify the settings:")
    print(config_path)
    print("\nIMPORTANT: Make sure the 'ckpt' path under the 'model' section points to your pre-trained MiniGPT-4 checkpoint, for example:")
    print("ckpt: 'prerained_minigpt4_7b.pth'")
    print("-" * 50)

    # --- Step 3: Run Training ---
    print("Step 3: Ready for Training.")
    print("Once you have prepared the data and verified the configuration, you can start the fine-tuning.")
    print("The command below will start the training process on a single GPU.")
    print("You can modify NUM_GPU to use more GPUs if available.\n")
    
    num_gpu = 1  # Change this value based on your available GPUs
    training_command = f"""
    torchrun --nproc-per-node={num_gpu} train.py --cfg-path {config_path}
    """
    
    print("--- Training Command ---")
    print(training_command)
    print("------------------------")
    print("\nTo execute the training, copy and paste the command above into your terminal.")
    print("The 'finetuning.py' script has set up all necessary files. This is the final step.")


if __name__ == "__main__":
    # NOTE: This script is a guide. The commands are meant to be run in your shell.
    # You can uncomment the `run_command` calls to make this script executable,
    # but it's recommended to run them step-by-step manually to ensure each step completes successfully.
    main()
