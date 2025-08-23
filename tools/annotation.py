# This program will the following things:
# - Create a new folder in output folder for annotation and data files
# - Extract IMU, SLAM and eyetracking data
# - Automatically name the folder and all files
# 
# IMPORTANT: Change file path for vrs, mp4 an output file locations

import json
import os
import shutil
from pathlib import Path
from datetime import datetime
import subprocess
import sys
import cv2
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import messagebox
import threading

# Test case configurations
TEST_CASES = {
    1: {"name": "TC01_object_localization", "description": "Object Localization"},
    2: {"name": "TC02_hzd_detection", "description": "Hazard Detection"},
    3: {"name": "TC03_scene_description", "description": "Scene Description"},
    4: {"name": "TC04_navigation", "description": "Navigation"},
    5: {"name": "TC05_social_cues", "description": "Social Cues"},
    6: {"name": "TC06_distance_est", "description": "Distance Estimation"},
    7: {"name": "TC07_task_instruction", "description": "Task Instruction"},
    8: {"name": "TC08_object_query", "description": "Object Query"},
    9: {"name": "TC09_txt_understanding", "description": "Text Understanding"},
    10: {"name": "TC10_motion_understanding", "description": "Motion Understanding"}
}

class VideoPreview:
    def __init__(self, video_path):
        self.video_path = video_path
        self.window = None
        
    def extract_middle_frame(self):
        """Extract the middle frame from the video"""
        try:
            cap = cv2.VideoCapture(str(self.video_path))
            if not cap.isOpened():
                print(f"Error: Cannot open video {self.video_path}")
                return None
                
            # Get total frame count
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            middle_frame_num = total_frames // 2
            
            # Set frame position to middle
            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_num)
            
            # Read the frame
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                # Convert BGR to RGB for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return frame_rgb
            else:
                print("Error: Could not read middle frame")
                return None
                
        except Exception as e:
            print(f"Error extracting frame: {e}")
            return None
    
    def show_preview(self):
        """Show video preview in a popup window"""
        frame = self.extract_middle_frame()
        if frame is None:
            messagebox.showerror("Error", f"Could not extract preview from {self.video_path.name}")
            return
            
        # Create Tkinter window
        self.window = tk.Toplevel()
        self.window.title(f"Video Preview: {self.video_path.name}")
        self.window.geometry("800x600")
        
        # Convert frame to PIL Image and resize for display
        pil_image = Image.fromarray(frame)
        
        # Calculate display size while maintaining aspect ratio
        display_width = 760
        display_height = 540
        img_width, img_height = pil_image.size
        
        # Calculate scaling factor
        scale_w = display_width / img_width
        scale_h = display_height / img_height
        scale = min(scale_w, scale_h)
        
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage for Tkinter
        from PIL import ImageTk
        photo = ImageTk.PhotoImage(pil_image)
        
        # Create and pack label with image
        label = tk.Label(self.window, image=photo)
        label.image = photo  # Keep a reference
        label.pack(expand=True)
        
        # Add close button
        close_btn = tk.Button(self.window, text="Close Preview", 
                             command=self.close_preview, font=("Arial", 12))
        close_btn.pack(pady=10)
        
        # Center the window
        self.window.update_idletasks()
        x = (self.window.winfo_screenwidth() // 2) - (self.window.winfo_width() // 2)
        y = (self.window.winfo_screenheight() // 2) - (self.window.winfo_height() // 2)
        self.window.geometry(f"+{x}+{y}")
        
        # Make window stay on top
        self.window.lift()
        self.window.attributes('-topmost', True)
        
        print(f"✓ Video preview opened for {self.video_path.name}")
        print("  (Close the preview window when done reviewing)")
    
    def close_preview(self):
        """Close the preview window"""
        if self.window:
            self.window.destroy()
            self.window = None

class VISTAProcessor:
    def __init__(self, vrs_folder, mp4_folder, output_base):
        self.vrs_folder = Path(vrs_folder)
        self.mp4_folder = Path(mp4_folder)
        self.output_base = Path(output_base)
        
        # Initialize Tkinter root (hidden)
        self.root = tk.Tk()
        self.root.withdraw()
        
    def search_vrs_file(self, vrs_filename):
        """Search for VRS file in the VRS folder"""
        # Try exact match first
        vrs_file = self.vrs_folder / vrs_filename
        if vrs_file.exists():
            return vrs_file
            
        # If no extension provided, try adding .vrs
        if not vrs_filename.endswith('.vrs'):
            vrs_file = self.vrs_folder / f"{vrs_filename}.vrs"
            if vrs_file.exists():
                return vrs_file
        
        # Search for partial matches
        matches = []
        search_stem = Path(vrs_filename).stem.lower()
        
        for vrs_candidate in self.vrs_folder.glob("*.vrs"):
            if search_stem in vrs_candidate.stem.lower():
                matches.append(vrs_candidate)
        
        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            print(f"\nMultiple VRS files found matching '{vrs_filename}':")
            for i, match in enumerate(matches, 1):
                print(f"  {i}: {match.name}")
            
            while True:
                try:
                    choice = int(input("Select file number: ")) - 1
                    if 0 <= choice < len(matches):
                        return matches[choice]
                    print("Invalid selection.")
                except ValueError:
                    print("Please enter a valid number.")
        
        return None
        
    def find_matching_mp4(self, vrs_file):
        """Find corresponding MP4 file for VRS file"""
        vrs_stem = vrs_file.stem
        
        # Try exact match first
        mp4_file = self.mp4_folder / f"{vrs_stem}.mp4"
        if mp4_file.exists():
            return mp4_file
            
        # Try common naming patterns
        matches = []
        for mp4_candidate in self.mp4_folder.glob("*.mp4"):
            if (vrs_stem.lower() in mp4_candidate.stem.lower() or 
                mp4_candidate.stem.lower() in vrs_stem.lower()):
                matches.append(mp4_candidate)
        
        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            print(f"\nMultiple MP4 files found matching '{vrs_file.name}':")
            for i, match in enumerate(matches, 1):
                print(f"  {i}: {match.name}")
            
            while True:
                try:
                    choice = int(input("Select file number: ")) - 1
                    if 0 <= choice < len(matches):
                        return matches[choice]
                    print("Invalid selection.")
                except ValueError:
                    print("Please enter a valid number.")
                    
        return None
    
    def extract_vrs_data(self, vrs_file, output_folder):
        """Extract data from VRS file using vrs command line tools"""
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        
        try:
            # Extract timestamp data
            timestamp_file = output_folder / "timestamps.csv"
            subprocess.run([
                "vrs", "extract-timestamps", str(vrs_file), 
                "--output", str(timestamp_file)
            ], check=True)
            
            # Extract IMU data
            imu_file = output_folder / "imu_data.csv"
            subprocess.run([
                "vrs", "extract-imu", str(vrs_file),
                "--output", str(imu_file)
            ], check=True)
            
            # Extract SLAM data
            slam_file = output_folder / "slam_data.json"
            subprocess.run([
                "vrs", "extract-slam", str(vrs_file),
                "--output", str(slam_file)
            ], check=True)
            
            # Extract eye tracking data
            eye_tracking_file = output_folder / "eye_tracking.csv"
            subprocess.run([
                "vrs", "extract-eyetracking", str(vrs_file),
                "--output", str(eye_tracking_file)
            ], check=True)
            
            print(f"✓ Successfully extracted VRS data to {output_folder}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"⚠ Warning: Failed to extract some VRS data: {e}")
            return False
        except FileNotFoundError:
            print("⚠ Warning: VRS command line tools not found. Skipping VRS extraction.")
            return False
    
    def save_preview_frame(self, mp4_file, output_folder):
        """Save the middle frame as a preview image"""
        try:
            preview = VideoPreview(mp4_file)
            frame = preview.extract_middle_frame()
            
            if frame is not None:
                # Save as PNG
                preview_path = output_folder / "preview_frame.png"
                pil_image = Image.fromarray(frame)
                pil_image.save(preview_path)
                print(f"✓ Saved preview frame: {preview_path}")
                return preview_path
            
        except Exception as e:
            print(f"Warning: Could not save preview frame: {e}")
        
        return None
    
    def get_user_metadata(self, vrs_file, mp4_file, test_case_info):
        """Prompt user for metadata input with video preview"""
        print(f"\n{'='*60}")
        print(f"METADATA INPUT")
        print(f"VRS File: {vrs_file.name}")
        print(f"MP4 File: {mp4_file.name}")
        print(f"Test Case: {test_case_info['description']}")
        print(f"{'='*60}")
        
        # Show video preview
        show_preview = input("\nShow video preview? (y/n): ").strip().lower()
        preview_window = None
        
        if show_preview == 'y':
            try:
                preview_window = VideoPreview(mp4_file)
                # Show preview in a separate thread to avoid blocking
                preview_thread = threading.Thread(target=preview_window.show_preview)
                preview_thread.daemon = True
                preview_thread.start()
            except Exception as e:
                print(f"Could not show preview: {e}")
        
        metadata = {}
        
        # Basic required fields
        print("\nPlease fill in the annotation details:")
        metadata["setup_description"] = input("Setup description: ").strip()
        metadata["user_query"] = input("User query: ").strip()
        metadata["descriptive_ground_truth"] = input("Descriptive ground truth: ").strip()
        metadata["action_ground_truth"] = input("Action ground truth: ").strip()
        
        # Environment settings
        print("\nEnvironment Settings:")
        metadata["environment"] = input("Environment (indoor/outdoor): ").strip() or "indoor"
        metadata["lighting"] = input("Lighting conditions: ").strip() or "fluorescent"
        
        # Technical parameters
        print("\nTechnical Parameters:")
        fps_input = input("FPS (default: 10): ").strip()
        metadata["fps"] = int(fps_input) if fps_input else 10
        
        resolution_input = input("Camera resolution (width,height) or default [1408,1408]: ").strip()
        if resolution_input:
            try:
                w, h = map(int, resolution_input.split(','))
                metadata["camera_resolution"] = [w, h]
            except:
                metadata["camera_resolution"] = [1408, 1408]
        else:
            metadata["camera_resolution"] = [1408, 1408]
            
        distance_input = input("Distance to target (meters): ").strip()
        if distance_input:
            try:
                metadata["distance_to_target"] = float(distance_input)
            except:
                metadata["distance_to_target"] = None
        
        # Optional fields
        print("\nOptional Fields:")
        measurable_result = input("Measurable result (optional): ").strip()
        if measurable_result:
            metadata["measurable_result"] = measurable_result
        
        # Close preview window if it was opened
        if preview_window and preview_window.window:
            preview_window.close_preview()
            
        return metadata
    
    def create_annotation_json(self, test_case_id, metadata, output_folder):
        """Create annotation JSON file"""
        annotation_data = {
            "id": test_case_id,
            "dataset_version": 1.0,
            "collection_date": datetime.now().strftime("%Y-%m-%d"),
            "test_case": test_case_id.split('_')[0],
            "task_type": TEST_CASES[int(test_case_id[2:4])]["name"].split('_', 1)[1],
            **metadata,
            "annotations": [
                {
                    "frame_id": 0,
                    "timestamp": 0.0,
                    "collection_date": datetime.now().strftime("%Y-%m-%d"),
                    **metadata
                }
            ]
        }
        
        json_file = output_folder / "annotations.json"
        with open(json_file, 'w') as f:
            json.dump(annotation_data, f, indent=2)
            
        print(f"✓ Created annotation file: {json_file}")
        return json_file
    
    def process_annotation(self):
        """Process a single annotation starting with VRS file search"""
        print(f"\n{'='*80}")
        print("NEW ANNOTATION SESSION")
        print(f"{'='*80}")
        
        # Get VRS file name from user
        while True:
            vrs_filename = input("\nEnter VRS file name (or 'quit' to exit): ").strip()
            
            if vrs_filename.lower() == 'quit':
                return 'quit'
                
            if not vrs_filename:
                print("Please enter a VRS file name.")
                continue
                
            # Search for VRS file
            vrs_file = self.search_vrs_file(vrs_filename)
            if vrs_file:
                print(f"✓ Found VRS file: {vrs_file.name}")
                break
            else:
                print(f"✗ VRS file '{vrs_filename}' not found.")
                retry = input("Try again? (y/n): ").strip().lower()
                if retry != 'y':
                    return 'continue'  # Continue to next annotation but don't quit
        
        # Find matching MP4 file
        mp4_file = self.find_matching_mp4(vrs_file)
        if not mp4_file:
            print(f"✗ No matching MP4 file found for {vrs_file.name}")
            return 'continue'  # Continue to next annotation
            
        print(f"✓ Found matching MP4 file: {mp4_file.name}")
        
        # Get test case selection
        print("\nAvailable Test Cases:")
        for num, info in TEST_CASES.items():
            print(f"  {num}: {info['description']}")
        
        while True:
            try:
                case_num = int(input("\nSelect test case number (1-10): "))
                if case_num in TEST_CASES:
                    break
                print("Invalid case number. Please select 1-10.")
            except ValueError:
                print("Please enter a valid number.")
        
        # Get specific ID
        while True:
            specific_id = input("Input specific ID (000-999): ").strip()
            if len(specific_id) == 3 and specific_id.isdigit():
                break
            print("Please enter a 3-digit ID (000-999).")
        
        # Create test case ID and folder structure
        test_case_id = f"TC{case_num:02d}_{specific_id}"
        test_case_info = TEST_CASES[case_num]
        
        # Check if folder already exists
        output_folder = self.output_base / test_case_info["name"] / test_case_id
        if output_folder.exists():
            overwrite = input(f"Folder {test_case_id} already exists. Overwrite? (y/n): ").strip().lower()
            if overwrite != 'y':
                print("Skipping this annotation.")
                return 'continue'
            shutil.rmtree(output_folder)
        
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # Copy files
        dest_mp4 = output_folder / mp4_file.name
        dest_vrs = output_folder / vrs_file.name
        
        shutil.copy2(mp4_file, dest_mp4)
        shutil.copy2(vrs_file, dest_vrs)
        
        print(f"Copied MP4 to: {dest_mp4}")
        print(f"Copied VRS to: {dest_vrs}")
        
        # Extract VRS data
        vrs_data_folder = output_folder / "vrs_data"
        self.extract_vrs_data(vrs_file, vrs_data_folder)
        
        # Save preview frame
        self.save_preview_frame(mp4_file, output_folder)
        
        # Get metadata from user (with video preview option)
        metadata = self.get_user_metadata(vrs_file, mp4_file, test_case_info)
        
        # Create annotation JSON
        self.create_annotation_json(test_case_id, metadata, output_folder)
        
        print(f"\nSuccessfully created annotation: {test_case_id}")
        print(f"  Output folder: {output_folder}")
        
        return 'success'

def main():
    print("VISTA Dataset Processing Tool")
    print("Single Annotation Mode with Video Preview")
    print("=" * 50)
    
    # Check if required libraries are available
    try:
        import cv2
        import PIL
        import tkinter
    except ImportError as e:
        print(f"Error: Missing required library: {e}")
        print("Please install required packages:")
        print("pip install opencv-python pillow")
        return
    
    # Get folder paths
    vrs_folder = "/Users/joshuayeh/dataset_project/hugging_face/VISTA/raw" #input("Enter VRS files folder path: ").strip()
    mp4_folder = "/Users/joshuayeh/dataset_project/hugging_face/VISTA/mp4" #input("Enter MP4 files folder path: ").strip()
    output_base = "/Users/joshuayeh/dataset_project/hugging_face/VISTA/annotations" #input("Enter output base folder path: ").strip()
    
    if not all([Path(p).exists() for p in [vrs_folder, mp4_folder]]):
        print("Error: One or more input folders don't exist.")
        return
    
    # Create output base if it doesn't exist
    Path(output_base).mkdir(parents=True, exist_ok=True)
    
    processor = VISTAProcessor(vrs_folder, mp4_folder, output_base)
    
    print(f"\nVRS Folder: {vrs_folder}")
    print(f"MP4 Folder: {mp4_folder}")
    print(f"Output Folder: {output_base}")
    print("\nNote: Video previews will show the middle frame of each video")
    
    # Process annotations one at a time
    annotations_created = 0
    
    while True:
        try:
            result = processor.process_annotation()
            
            if result == 'quit':
                break
            elif result == 'success':
                annotations_created += 1
            
            # Ask if user wants to continue
            print(f"\nOptions:")
            print("1. Create another annotation")
            print("2. Exit program")
            
            choice = input("Enter choice (1 or 2): ").strip()
            
            if choice == '2':
                break
            elif choice != '1':
                print("Invalid choice. Exiting.")
                break
                
        except KeyboardInterrupt:
            print("\n\nProcessing interrupted by user.")
            break
        except Exception as e:
            print(f"\nError during annotation: {e}")
            continue_choice = input("Continue with next annotation? (y/n): ").strip().lower()
            if continue_choice != 'y':
                break
    
    print(f"\n{'='*80}")
    print("SESSION COMPLETE")
    print(f"{'='*80}")
    print(f"Total annotations created: {annotations_created}")
    print("Exit.")

if __name__ == "__main__":
    main()